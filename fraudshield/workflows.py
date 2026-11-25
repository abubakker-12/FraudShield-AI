"""Case, batch and system-health pages."""
import hashlib
import json
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from .batch import read_upload
from .dashboard import AMBER, GREEN, RED, chart, metrics, report_panel
from .features import FIELDS, ROOT
from .glass import band
from .presentation import label
from .reports import provider_model
from .storage import STATUSES


def result_table(rows, selectable=False, key=None):
    frame = pd.DataFrame(rows).drop(columns=["payload", "notes"], errors="ignore")
    for field in ("analyzed_at", "updated_at", "changed_at"):
        if field in frame:
            frame[field] = pd.to_datetime(frame[field], utc=True, format="mixed").dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    if "suspected" in frame:
        frame["suspected"] = frame["suspected"].map({1: "Suspected fraud", 0: "Likely legitimate"})
    table = frame
    if "status" in frame:
        colors = {"Pending review": AMBER, "Under investigation": "#77B8FF",
                  "Confirmed fraud": RED, "Dismissed": GREEN}
        table = frame.style.map(lambda value: f"color: {colors.get(value, '#E7EFFC')}", subset=["status"])
    selection = dict(on_select="rerun", selection_mode="single-row", key=key) if selectable else {}
    return st.dataframe(table, hide_index=True, width="stretch", column_config={
        "id": "Record ID", "transaction_id": "Transaction ID", "model_version": "Model version",
        "suspected": "Model prediction", "risk": st.column_config.NumberColumn("Model score", format="percent"),
        "threshold": st.column_config.NumberColumn("Threshold", format="percent"), "provider": "Explanation provider",
        "analyzed_at": "Analyzed at", "updated_at": "Reviewed at", "status": "Human review status",
    }, **selection)


def safe_csv(rows):
    def escape(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
    return pd.DataFrame(rows).map(escape).to_csv(index=False)


def saved(store):
    st.header("Saved Fraud Cases")
    stats, _ = store.activity()
    band("Review queue", "layers", "Human review outcomes and outstanding cases", [
        ("Suspected cases", f"{stats['cases']:,}", "warning", "rose"),
        ("Pending review", f"{stats['pending']:,}", "pending", "amber"),
        ("Human-confirmed", f"{stats['confirmed']:,}", "shield", "teal"),
        ("Dismissed", f"{stats['dismissed']:,}", "file", "muted"),
    ])
    cases, history = st.tabs(["Fraud cases", "Prediction history"])
    with cases:
        with st.container(key="fs_panel_case_filters"):
            a, b, c = st.columns(3)
            query = a.text_input("Search saved cases", max_chars=80)
            state = b.selectbox("Review status", ["All", *STATUSES])
            sort = c.selectbox("Order cases", ["recent", "high", "low"],
                               format_func={"recent": "Most recent", "high": "Highest score", "low": "Lowest score"}.get)
            page = st.number_input("Cases page", min_value=1, step=1)
        records = store.records(query, None if state == "All" else state, 100, (page - 1) * 100, sort=sort)
        if not records:
            st.info("No saved fraud cases on this page.")
        else:
            event = result_table(records, selectable=True, key=f"case_rows_{query}_{state}_{sort}_{page}")
            selected = event.selection.rows
            st.download_button("Export case page CSV", safe_csv(records), "fraud-cases.csv", "text/csv")
            focused = st.selectbox("Case to review", records, index=selected[0] if selected and selected[0] < len(records) else 0,
                                key=f"case_picker_{selected[0] if selected else 'none'}_{query}_{state}_{sort}_{page}",
                                format_func=lambda row: f"#{row['id']} · {row['transaction_id']} · {row['status']}")
            case = focused
            with st.form(f"case_review_{case['id']}"):
                outcome = st.selectbox("Set review status", STATUSES, index=STATUSES.index(case["status"]))
                notes = st.text_area("Reviewer notes", value=case["notes"] or "", max_chars=1500)
                if st.form_submit_button("Save human review", type="primary"):
                    store.review(case["id"], outcome, notes)
                    st.rerun()
            audit = store.review_history(case["id"])
            payload = json.loads(case["payload"])
            with st.expander("Saved investigation report"):
                report_panel(payload["explanation"])
            with st.expander("Review history"):
                for event in audit:
                    transition = f"{event['prior_status']} → {event['status']}" if event["prior_status"] else event["status"]
                    st.html(f'<div class="fs-timeline"><time>{escape(event["changed_at"])}</time>'
                            f'<strong>{escape(transition)}</strong><p>{escape(event["notes"] or "")}</p></div>')
            export = {**case, "payload": payload, "review_history": audit}
            st.download_button("Export individual case", json.dumps(export, indent=2), f"case-{case['id']}.json",
                               "application/json")
    with history:
        query = st.text_input("Search prediction history", max_chars=80)
        page = st.number_input("History page", min_value=1, step=1)
        rows = store.records(query, limit=100, offset=(page - 1) * 100, history=True)
        if rows:
            result_table(rows)
            case = st.selectbox("Previous analysis", rows, format_func=lambda row: f"#{row['id']} · {row['transaction_id']}")
            with st.expander("Previous investigation report"):
                report_panel(json.loads(case["payload"])["explanation"])
        else:
            st.info("No predictions on this page.")


@st.cache_data(max_entries=16)
def example_csv():
    return pd.read_csv(ROOT / "data" / "demo.csv", nrows=25)[FIELDS].to_csv(index=False)


def batch(service, provider, public):
    st.header("Batch Prediction")
    st.caption("CSV · UTF-8 · Up to 1,000 rows and 2 MB. Open a flagged row to generate its report.")
    with st.container(key="fs_panel_upload"):
        a, b = st.columns(2)
        a.download_button("Download CSV template", ",".join(FIELDS) + "\n", "batch-template.csv", "text/csv")
        b.download_button("Download synthetic examples", example_csv(), "batch-examples.csv", "text/csv")
        with st.expander("Required columns"):
            st.code("\n".join(f"{field:<25} {label(field)}" for field in FIELDS), language=None)
        upload = st.file_uploader("Transaction CSV", type=["csv"])
    if not upload:
        return
    try:
        if upload.size > 2_000_000:
            raise ValueError("Upload a CSV smaller than 2 MB.")
        raw = upload.getvalue()
        key = hashlib.sha256(raw).hexdigest()
        if st.session_state.get("validated_batch_key") != key:
            st.session_state.validated_batch = read_upload(raw)
            st.session_state.validated_batch_key = key
        frame = st.session_state.validated_batch
    except ValueError as error:
        st.error(str(error))
        return
    with st.expander("Validated data preview"):
        st.dataframe(frame.head(10).rename(columns=label), hide_index=True, width="stretch")
    if st.button("Score batch", type="primary"):
        progress = st.progress(0, text="Scoring batch")
        scores = []
        for start in range(0, len(frame), 250):
            scores.extend(service.scorer.batch(frame.iloc[start:start+250]))
            progress.progress(len(scores) / len(frame), text=f"Scored {len(scores):,} of {len(frame):,}")
        st.session_state.batch_result = (key, pd.DataFrame(scores))
    previous = st.session_state.get("batch_result")
    if not previous or previous[0] != key:
        return
    result = previous[1].copy()
    result.insert(0, "Source record", result.index + 1)
    result["Prediction"] = result.suspected_fraud.map({True: "Suspected fraud", False: "Likely legitimate"})
    flagged = result[result.suspected_fraud]
    metrics([("Rows scored", len(result)), ("Suspected fraud", len(flagged)), ("Likely legitimate", len(result)-len(flagged))])
    chart(px.histogram(result, x="risk", color="Prediction", title="Batch score distribution",
                       labels={"risk": "Model score"},
                       color_discrete_map={"Suspected fraud": RED, "Likely legitimate": GREEN}).update_xaxes(tickformat=".0%"))
    choice = st.selectbox("Show predictions", ["All", "Suspected fraud", "Likely legitimate"])
    visible = result if choice == "All" else result[result.Prediction.eq(choice)]
    st.dataframe(visible.drop(columns="suspected_fraud"), hide_index=True, width="stretch",
                 column_config={"risk": st.column_config.NumberColumn("Model score", format="percent"),
                                "threshold": st.column_config.NumberColumn("Review threshold", format="percent"),
                                "model_version": "Model version"})
    st.download_button("Download predictions", safe_csv(result.to_dict("records")), "batch-predictions.csv", "text/csv")
    if len(flagged):
        row_number = st.selectbox("Flagged record to investigate", flagged["Source record"].tolist())
        if public:
            st.caption("Uploaded transaction reports use the offline explanation engine in the public workspace.")
        if st.button("Open in investigation workspace"):
            with st.spinner("Preparing an individual report…"):
                transaction_id = f"BATCH-{key[:20]}-{row_number}"
                st.session_state.analysis = service.analyze_row(
                    transaction_id, frame.iloc[row_number-1].to_dict(), "python" if public else provider)
            st.session_state.navigate = "Fraud Investigation"
            st.rerun()


def settings(service, provider, available, labels, public):
    st.header("Settings")
    model = service.scorer.metadata
    metrics([("Active classifier", model["model"]), ("Review threshold", f"{model['threshold']:.2%}"),
             ("Database", "Connected")])
    st.caption(f"Version {model['version']} · Model artifacts loaded · "
               f"{'Public synthetic session' if public else 'Local workspace'}")
    result = st.session_state.get("analysis")
    if result and "inference_ms" in result:
        st.caption(f"Most recent model inference {result['inference_ms']:.1f} ms")
    with st.container(key="fs_panel_provider_config"):
        selected = st.selectbox("Explanation provider", available, index=available.index(provider), format_func=labels.get)
        st.session_state.llm_provider = selected
        engine = service.engine
        st.info(f"{labels[selected]} · {provider_model(engine, selected) or 'Model ID required'} · {engine.availability(selected)}")
        st.caption("The Python explanation engine is available offline.")
    with st.container(key="fs_panel_providers"):
        st.subheader("Provider and storage status")
        st.caption(f"Groq Cloud · {engine.availability('groq')}")
        st.caption(f"Local Ollama · {engine.availability('ollama') if not public else 'Unavailable in public demo'}")
        st.caption(f"SQLite · {'Temporary, session-isolated' if public else 'Persistent local workspace'}")
        st.caption("Synthetic demo catalog" if service.catalog.synthetic else "Private local transaction catalog")
    if st.button("Test provider connection"):
        with st.spinner("Checking the configured provider…"):
            connected = engine.test_connection(selected)
        if connected:
            st.success("Provider is reachable and the configured model is available.")
        else:
            st.error("Provider check failed. Review the model configuration or connection.")
