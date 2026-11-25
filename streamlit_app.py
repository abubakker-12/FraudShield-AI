import os
import sqlite3
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.express as px
import streamlit as st

from fraudshield.explanations import ExplanationEngine
from fraudshield.features import RAW, ROOT
from fraudshield.glass import brand, contact_footer, install, topbar
from fraudshield.service import Catalog, InvestigationService, ModelUnavailable, Scorer
from fraudshield.dashboard import (NAVIGATION, chart, copilot_panel, details, evidence_charts,
                                   explorer, metrics, overview, report_panel, risk_chart)
from fraudshield.storage import Store
from fraudshield.workflows import batch, saved, settings

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")


def setting(key, default=""):
    try:
        return str(st.secrets.get(key, os.getenv(key, default)))
    except (FileNotFoundError, KeyError):
        return os.getenv(key, default)


@st.cache_resource
def resources(use_local):
    path = RAW if use_local else ROOT / "data" / "demo.csv"
    keys = ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
            "LLM_TIMEOUT_SECONDS", "OLLAMA_TIMEOUT_SECONDS", "LLM_REQUESTS_PER_MINUTE")
    return Catalog(path), Scorer(), ExplanationEngine({key: setting(key) for key in keys})


def investigation(engine, provider, public, service=None):
    st.header("Fraud Investigation")
    result = st.session_state.get("analysis")
    if result is None:
        st.info("Select a transaction in Transaction Explorer and run an analysis.")
        return
    if result["suspected_fraud"]:
        st.error(f"{result['transaction_id']} · Suspected fraud")
    else:
        st.success(f"{result['transaction_id']} · Likely legitimate")
    metrics([("Uncalibrated model score", f"{result['risk']:.2%}"),
             ("Review threshold", f"{result['threshold']:.2%}"),
             ("SHAP reference score", f"{result['base_score']:.2%}")])
    st.caption(f"{result['model_version']} | Uncalibrated model score | {result.get('analyzed_at', 'Time unavailable')}")
    if "inference_ms" in result:
        st.caption(f"Measured preprocessing and inference {result['inference_ms']:.1f} ms")
    with st.container(key="fs_panel_score"):
        risk_chart(result)
    if result.get("transaction"):
        details(result["transaction"])
    with st.container(key="fs_panel_evidence"):
        evidence_charts(result)
    with st.container(key="fs_panel_report"):
        report_panel(result["explanation"])
    with st.container(key="fs_panel_copilot"):
        copilot_panel(result, engine, "python" if public and result["transaction_id"].startswith("BATCH-") else provider,
                      service)


def performance(m):
    st.header("Model Performance")
    st.caption(f"{m['model']} · {m['version']} · Chronological, customer-disjoint evaluation")
    test = m["test"]
    metrics([("Average precision", f"{test['average_precision']:.3f}"), ("ROC AUC", f"{test['roc_auc']:.3f}"),
             ("Precision", f"{test['precision']:.2%}"), ("Recall", f"{test['recall']:.2%}")])
    metrics([("F1 score", f"{test['f1']:.3f}"), ("Accuracy", f"{test['accuracy']:.2%}"),
             ("False-positive rate", f"{test['false_positive_rate']:.2%}"),
             ("False-negative rate", f"{test['false_negative_rate']:.2%}")])
    st.subheader("Validation comparison")
    st.dataframe(pd.DataFrame(m["validation"]).T.drop(columns="confusion_matrix"), width="stretch")
    with st.expander("Held-out test metrics"):
        st.json(test)
    a, b = st.columns(2)
    with a:
        chart(px.line(pd.DataFrame(m["roc"]), x="FPR", y="TPR", title="Held-out ROC"))
        chart(px.imshow(test["confusion_matrix"], x=["Likely legitimate", "Suspected fraud"],
                            y=["Legitimate", "Fraud"], text_auto=True, color_continuous_scale="Blues",
                            labels={"x": "Predicted", "y": "Recorded"}, title="Held-out confusion matrix"))
    importance = pd.Series(m["importance"], name="Mean absolute SHAP").nlargest(12).sort_values()
    with b:
        chart(px.line(pd.DataFrame(m["pr"]), x="Recall", y="Precision", title="Held-out precision–recall"))
        chart(px.bar(importance, orientation="h", title="Global feature importance"))
    with st.expander("Development model comparison · 2.0"):
        from fraudshield.dashboard import experiment_report
        report = experiment_report()
        rows = {name: result["balanced"] for name, result in report["candidates"].items()}
        rows["Active model, same development rows"] = report["baseline"]
        st.dataframe(pd.DataFrame(rows).T[["average_precision", "roc_auc", "f1", "f2", "brier"]],
                     width="stretch")
        st.caption("Challengers were ranked on development data. The existing model and threshold were retained.")
    st.caption("Scores are uncalibrated. Customer-disjoint performance reflects this source and time period. "
               "Feature availability assumptions and real-world transfer need independent validation.")


def main():
    install()
    use_local = setting("FRAUDSHIELD_USE_LOCAL_DATA") == "1" and RAW.exists()
    public = setting("FRAUDSHIELD_PUBLIC_DEMO") == "1"
    try:
        catalog, scorer, engine = resources(use_local and not public)
    except ModelUnavailable:
        st.error("Required demo data or model artifacts are missing. See the README for setup.")
        return
    if public:
        if "demo_directory" not in st.session_state:
            st.session_state.demo_directory = TemporaryDirectory(prefix="fraudshield-")
        db = os.path.join(st.session_state.demo_directory.name, "demo.db")
    else:
        db = ROOT / setting("DATABASE_PATH", "database/fraudshield.db")
    try:
        if st.session_state.get("workspace_db") != str(db):
            st.session_state.workspace_store = Store(db)
            st.session_state.workspace_db = str(db)
        service = InvestigationService(catalog, scorer, st.session_state.workspace_store, engine)
    except (sqlite3.Error, OSError):
        st.error("The investigation database is unavailable. Check local storage permissions.")
        return
    labels = {"groq": "Groq Cloud", "ollama": "Local Ollama", "python": "Python fallback"}
    available = ["groq", "python"] if public else ["groq", "ollama", "python"]
    default = setting("LLM_PROVIDER", "groq")
    brand()
    if "navigate" in st.session_state:
        st.session_state.page = st.session_state.pop("navigate")
    page = st.sidebar.radio("Workspace", list(NAVIGATION), key="page",
                            label_visibility="collapsed", width="stretch",
                            format_func=lambda name: f"{NAVIGATION[name]}  {name}")
    provider = st.session_state.get("llm_provider", default if default in available else "python")
    status = engine.availability(provider)
    topbar(scorer.metadata["model"].split()[0], labels[provider], status,
           "Investigation Command Center" if page == "Overview" else page)
    st.sidebar.markdown("**System status**")
    st.sidebar.caption(f"Model ready | {scorer.metadata['model']}")
    st.sidebar.caption(f"{labels[provider]} | {status}")
    mode = "Synthetic demo" if catalog.synthetic else "Local source"
    st.sidebar.caption(mode)
    if page == "Settings":
        settings(service, provider, available, labels, public)
    else:
        actions = {
            "Overview": lambda: overview(service.store, mode), "Transaction Explorer": lambda: explorer(service, provider),
            "Fraud Investigation": lambda: investigation(engine, provider, public, service), "Saved Fraud Cases": lambda: saved(service.store),
            "Model Performance": lambda: performance(scorer.metadata), "Batch Prediction": lambda: batch(service, provider, public),
        }
        try:
            actions[page]()
        except sqlite3.Error:
            st.error("The investigation database could not complete this operation. Try again.")
        except (OSError, KeyError, ValueError):
            st.error("The selected workspace data could not be processed. Check the input and try again.")
    contact_footer(
        setting("FRAUDSHIELD_CONTACT_NAME"),
        setting("FRAUDSHIELD_CONTACT_EMAIL"),
        setting("FRAUDSHIELD_CONTACT_PHONE"),
        setting("FRAUDSHIELD_CONTACT_LINKEDIN"),
    )


main()
