"""Presentation components for the Streamlit command center."""
import hashlib
import json
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .features import FIELDS, ROOT
from .glass import band, icon, panel_heading
from .presentation import display, label
from .service import selected_transaction
from .reports import QUESTIONS, copilot

BLUE, GREEN, AMBER, RED = "#00BFFF", "#2DD4BF", "#FBBF24", "#FB527B"
NAVIGATION = {
    "Overview": ":material/home:", "Transaction Explorer": ":material/search:",
    "Fraud Investigation": ":material/target:", "Saved Fraud Cases": ":material/layers:",
    "Model Performance": ":material/bar_chart:", "Batch Prediction": ":material/database:",
    "Settings": ":material/settings:",
}


def chart(fig, percent=False, compact=False):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Arial, sans-serif", color="#F1F5F9", size=12),
                      colorway=[BLUE, "#8B5CF6", GREEN, AMBER, RED],
                      margin=dict(l=12, r=18, b=26, t=12 if compact else 48),
                      legend=dict(orientation="h", y=-.24), height=233 if compact else 300)
    fig.update_xaxes(gridcolor="#88aaec24", zerolinecolor="#88aaec45", automargin=True)
    fig.update_yaxes(gridcolor="#88aaec24", zerolinecolor="#88aaec45", automargin=True)
    if percent:
        fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "scrollZoom": False})


def metrics(items):
    cols = st.columns(len(items))
    for col, (name, value) in zip(cols, items):
        col.metric(name, value, border=True)


@st.cache_data
def intelligence():
    folder = ROOT / "reports"
    return tuple(json.loads((folder / file).read_text(encoding="utf-8"))
                 for file in ("eda.json", "intelligence.json"))


@st.cache_data
def experiment_report():
    return json.loads((ROOT / "reports" / "experiment-v2.json").read_text(encoding="utf-8"))


def overview(store, mode):
    eda, extra = intelligence()
    st.title("FraudShield AI")
    st.caption("Explainable Fraud Detection and Generative AI Investigation Platform")
    st.html('<div class="fs-accent" aria-hidden="true"></div>')
    # Preserve the native page heading for assistive technology.
    with st.container(key="fs_overview_heading"):
        st.header("Overview")
    stats, recent = store.activity()
    band("Investigation Workspace", "layers", "Model analyses and case review activity in this workspace", [
        ("Transactions analyzed", f"{stats['analyzed']:,}", "search", ""),
        ("Suspected fraud", f"{stats['cases']:,}", "warning", "rose"),
        ("Pending review", f"{stats['pending']:,}", "pending", "amber"),
        ("Human-confirmed", f"{stats['confirmed']:,}", "shield", "teal"),
        ("Dismissed", f"{stats['dismissed']:,}", "file", "muted"),
    ], tag=f"{mode}  |  Human review")
    band("Historical Dataset Intelligence", "data", "Labeled historical records used for training and evaluation", [
        ("Transactions", f"{eda['rows']:,}", "data", ""),
        ("Recorded fraud", f"{eda['fraud']:,}", "warning", "rose"),
        ("Recorded legitimate", f"{eda['rows'] - eda['fraud']:,}", "shield", "teal"),
        ("Recorded fraud rate", f"{eda['fraud_rate']:.2%}", "pie", "violet"),
    ], tag=f"Dataset period · {eda['dates'][0][:10]} to {eda['dates'][1][:10]}", historical=True)
    with st.container(key="fs_overview_grid"):
        a, b, c = st.columns([1.06, 1, 1.04], gap="small")
        with a, st.container(key="fs_panel_hour"):
            st.html(panel_heading("Hourly Fraud Rate", "chart"))
            hour = pd.DataFrame(eda["hourly"])
            fig = px.area(hour, x="hour_of_day", y="rate", labels={"hour_of_day": "Hour of day", "rate": "Recorded fraud rate"})
            fig.update_traces(mode="lines+markers", line=dict(color=BLUE, width=3),
                              fillcolor="rgba(0,191,255,.18)", marker=dict(color="#e7faff", size=5),
                              hovertemplate="Hour %{x}<br>Recorded rate %{y:.2%}<extra></extra>")
            # The focused scale makes hourly differences legible. The fill starts at the visible floor.
            fig.update_yaxes(range=[max(0, float(hour.rate.min()) - .004), float(hour.rate.max()) + .004])
            chart(fig, True, compact=True)
        with b, st.container(key="fs_panel_merchant"):
            st.html(panel_heading("Merchant Categories", "chart"))
            merchants = pd.DataFrame(eda["categories"]["merchant_category"]).nlargest(8, "rate").sort_values("rate")
            fig = px.bar(merchants, x="rate", y="merchant_category", orientation="h", text="rate",
                         labels={"rate": "Recorded fraud rate", "merchant_category": ""}, hover_data=["total", "fraud"])
            fig.update_traces(marker_color=BLUE, texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
            fig.update_xaxes(tickformat=".0%", range=[0, float(merchants.rate.max()) * 1.28])
            chart(fig, compact=True)
        with c, st.container(key="fs_panel_activity"):
            st.html(panel_heading("Recent Investigation Activity", "clock"))
            recent_activity(store.records(limit=5, history=True))
    amount = eda["amount"]
    st.caption(f"Historical amounts · Median {amount['50%']:,.2f} · Mean {amount['mean']:,.2f} · Maximum {amount['max']:,.2f}")
    distribution, categories, authentication, amounts = st.tabs(
        ["Recorded outcomes", "Merchant & payment", "Authentication & location", "Amounts"])
    with distribution:
        chart(px.pie(names=["Recorded legitimate", "Recorded fraud"],
                     values=[eda["rows"]-eda["fraud"], eda["fraud"]], hole=.68, title="Recorded outcomes",
                     color_discrete_sequence=[BLUE, RED]))
        if recent:
            chart(px.bar(pd.DataFrame(recent), x="day", y=["analyses", "flagged"], title="Recent workspace analyses",
                         barmode="group", color_discrete_sequence=[BLUE, RED],
                         labels={"day": "UTC day", "value": "Analyses", "variable": "Activity"}))
    with categories:
        for field in ("merchant_category", "payment_method"):
            rates = pd.DataFrame(eda["categories"][field]).sort_values("rate", ascending=False)
            chart(px.bar(rates, x=field, y="rate", title=f"Recorded fraud rate by {label(field).lower()}",
                         hover_data=["fraud", "total"], labels={field: label(field), "rate": "Fraud rate"}), True)
    with authentication:
        for field in ("failed_attempts", "is_international"):
            frame = pd.DataFrame(extra[field])
            frame[field] = frame[field].map(lambda x: display(field, x))
            chart(px.bar(frame, x=field, y="rate", title=f"Recorded fraud rate by {label(field).lower()}",
                         hover_data=["total", "fraud"], labels={field: label(field), "rate": "Fraud rate"}), True)
        frame = pd.DataFrame(eda["interactions"]["international_and_failures"])
        frame["Location"] = frame.is_international.map({0: "Domestic", 1: "International"})
        frame["Failed attempts"] = frame.repeat_failures.map({False: "0–1", True: "2 or more"})
        chart(px.bar(frame, x="Location", y="rate", color="Failed attempts", barmode="group",
                     title="Location and repeated failures", hover_data=["total", "fraud"],
                     labels={"rate": "Fraud rate"}, color_discrete_sequence=[BLUE, AMBER]), True)
    with amounts:
        frame = pd.DataFrame(extra["amount_distribution"])
        frame["Recorded outcome"] = frame.is_fraud.map({0: "Legitimate", 1: "Fraud"})
        frame["Share"] = frame.total / frame.groupby("Recorded outcome").total.transform("sum")
        chart(px.bar(frame, x="amount_band", y="Share", color="Recorded outcome", barmode="group",
                     title="Amount distribution within each recorded class", hover_data=["total"],
                     labels={"amount_band": "Amount band"}, color_discrete_sequence=[BLUE, RED]), True)
    st.subheader("Key fraud insights")
    row = next(r for r in eda["interactions"]["international_and_failures"]
               if r["is_international"] and r["repeat_failures"])
    st.write(f"International transactions with repeated failures had a **{row['rate']:.2%}** recorded fraud rate "
             f"({row['fraud']:,} of {row['total']:,} transactions).")
    effect = eda["statistical_comparisons"]["amount_probability_of_superiority"]
    st.write(f"A fraud amount exceeded a legitimate amount in an estimated **{effect:.2%}** of pairs, "
             "counting ties as half. Amount alone offered limited separation.")
    st.caption("The supplied dataset lacks verified production lineage. Historical associations describe this source cohort.")


def recent_activity(rows):
    if not rows:
        st.caption("Analyzed transactions will appear here.")
        return
    html = []
    for row in rows:
        when = pd.to_datetime(row["analyzed_at"], utc=True, errors="coerce")
        timestamp = when.strftime("%b %d, %H:%M UTC") if pd.notna(when) else "Time unavailable"
        flag = bool(row["suspected"])
        outcome = "Suspected fraud" if flag else "Likely legitimate"
        html.append(f'<div class="fs-recent-row">{icon("file")}'
                    f'<span class="fs-recent-id"><b>{escape(row["transaction_id"])}</b><time>{timestamp}</time></span>'
                    f'<span class="fs-state {"fraud" if flag else "ready"}">{outcome}</span>'
                    f'<b>{row["risk"]:.2%}</b></div>')
    st.html("".join(html))


def details(row):
    with st.container(border=True):
        cols = st.columns(3)
        for index, field in enumerate(FIELDS):
            if field in row:
                cols[index % 3].caption(label(field))
                cols[index % 3].html(f"<strong>{escape(display(field, row[field]))}</strong>")


@st.cache_data
def demo_examples(_scorer, version):
    source = pd.read_csv(ROOT / "data" / "demo.csv")
    scores = _scorer.batch(source)
    found = {}
    for tid, score in zip(source.transaction_id, scores):
        found.setdefault(score["suspected_fraud"], tid)
        if len(found) == 2:
            break
    return found


def explorer(service, provider):
    st.header("Transaction Explorer")
    st.caption("Synthetic demonstration records" if service.catalog.synthetic else "Private local source records")
    revision = st.session_state.get("filter_revision", 0)
    with st.container(key="fs_panel_filters"):
        if st.button("Clear filters"):
            st.session_state.filter_revision = revision + 1
            st.session_state.pop("shortcut", None)
            st.rerun()
        a, b, c, d = st.columns(4)
        query = a.text_input("Search transaction ID", max_chars=80, key=f"query_{revision}")
        location = b.selectbox("Location", ["All", "Domestic", "International"], key=f"location_{revision}")
        filter_values = {}
        for column, field in ((c, "merchant_category"), (d, "payment_method")):
            options = sorted(service.catalog.data[field].unique())
            filter_values[field] = column.selectbox(label(field), [None, *options], format_func=lambda x: x or "All",
                                                    key=f"{field}_{revision}")
        with st.expander("More filters and sorting"):
            a, b, c, d, e = st.columns(5)
            device = a.selectbox("Device Type", [None, *sorted(service.catalog.data.device_type.unique())],
                                 format_func=lambda x: x or "All", key=f"device_{revision}")
            minimum = b.number_input("Minimum amount", min_value=0., key=f"min_{revision}")
            maximum = c.number_input("Maximum amount (0 = any)", min_value=0., key=f"max_{revision}")
            failures = d.number_input("Minimum failed attempts", min_value=0, max_value=1000, key=f"failures_{revision}")
            sort = e.selectbox("Sort by", [None, "amount_high", "amount_low", "failures"],
                               format_func={None: "Default", "amount_high": "Amount high to low",
                                            "amount_low": "Amount low to high", "failures": "Most failures"}.get,
                               key=f"sort_{revision}")
    if service.catalog.synthetic:
        for col, (flag, tid) in zip(st.columns(2), demo_examples(service.scorer, service.scorer.metadata["version"]).items()):
            if col.button("Open suspected fraud example" if flag else "Open likely legitimate example"):
                st.session_state.shortcut = tid
    args = dict(query=query, international={"All": None, "Domestic": False, "International": True}[location],
                payment=filter_values["payment_method"], merchant=filter_values["merchant_category"],
                minimum=minimum, maximum=maximum or None, device=device, failures=failures, sort=sort)
    token = hashlib.sha256(repr((revision, args)).encode()).hexdigest()[:12]
    total, _ = service.catalog.browse(**args, limit=0)
    st.caption(f"{total:,} matching transactions")
    chosen = st.session_state.get("shortcut")
    if total:
        page_number = st.number_input("Page", 1, (total + 49) // 50, key=f"page_{token}")
        _, page = service.catalog.browse(**args, offset=(page_number-1)*50, limit=50)
        shown = page.copy()
        for field in ("is_international", "pin_changed_recently", "hour_of_day"):
            shown[field] = shown[field].map(lambda value: display(field, value))
        shown = shown.rename(columns=label)
        event = st.dataframe(shown, key=f"transactions_{token}_{page_number}", on_select="rerun",
                             selection_mode="single-row", hide_index=True, width="stretch",
                             column_config={"Transaction Amount": st.column_config.NumberColumn(format="%.2f"),
                                            "Time Since Previous Transaction": st.column_config.NumberColumn(format="%.2f h")})
        clicked = selected_transaction(page, event.selection.rows)
        chosen = clicked or chosen
    if chosen:
        st.subheader(chosen)
        details(service.catalog.get(chosen))
        if not service.catalog.synthetic and st.checkbox("Reveal recorded dataset outcome", key=f"actual_{chosen}"):
            st.write(service.catalog.actual(chosen))
        if st.button("Analyze Transaction", type="primary"):
            with st.spinner("Scoring the transaction and preparing its investigation…"):
                st.session_state.analysis = service.analyze(chosen, provider)
            st.session_state.navigate = "Fraud Investigation"
            st.rerun()


def risk_chart(result):
    score, threshold = result["risk"], result["threshold"]
    chart(go.Figure(go.Indicator(
        mode="gauge+number", value=100*score, number={"suffix": "%", "valueformat": ".2f"},
        title={"text": "Model score"}, gauge={"shape": "bullet", "axis": {"range": [0, 100]},
        "bar": {"color": RED if result["suspected_fraud"] else GREEN},
        "steps": [{"range": [0, 100*threshold], "color": "#183C38"},
                  {"range": [100*threshold, 100], "color": "#4D2638"}],
        "threshold": {"line": {"color": AMBER, "width": 4}, "value": 100*threshold}},
    )))


def evidence_charts(result):
    frame = pd.DataFrame(result["attributions"])
    frame["Feature"] = frame.feature.map(label)
    frame["Direction"] = frame.contribution.map(lambda x: "Raises score" if x > 0 else "Lowers score")
    top = frame.loc[frame.contribution.abs().gt(1e-9)].head(10)
    a, b = st.tabs(["Signed contributions", "Score reconstruction"])
    with a:
        chart(px.bar(top.iloc[::-1], y="Feature", x="contribution", color="Direction", orientation="h",
                     hover_data=[col for col in ["feature_value"] if col in top],
                     labels={"contribution": "Score change (%)"}, title="Largest SHAP contributions",
                     color_discrete_map={"Raises score": RED, "Lowers score": GREEN}).update_xaxes(tickformat="+.1%"))
    with b:
        tail = float(frame.contribution.sum() - top.contribution.sum())
        chart(go.Figure(go.Waterfall(x=["Reference", *top.Feature, "Other features", "Final score"],
                    y=[result["base_score"], *top.contribution, tail, 0],
                    measure=["absolute", *(["relative"]*len(top)), "relative", "total"],
                    increasing={"marker": {"color": RED}}, decreasing={"marker": {"color": GREEN}},
                    totals={"marker": {"color": BLUE}})).update_yaxes(tickformat=".0%"))
    with st.expander("Complete feature evidence"):
        st.dataframe(frame.drop(columns="feature"), hide_index=True, width="stretch",
                     column_config={"contribution": st.column_config.NumberColumn("SHAP contribution", format="percent"),
                                    "feature_value": st.column_config.TextColumn("Actual feature value")})


def report_panel(explanation):
    report = explanation.get("report")
    st.caption(f"{explanation['provider'].title()} · {explanation.get('provider_model', 'Recorded explanation')}")
    if not report:
        st.text(explanation["assessment"])
        st.text(explanation["reasoning"])
        for action in explanation["actions"]:
            st.text(action)
        return
    st.subheader("Executive summary")
    st.text(report["executive_summary"]["text"])
    st.caption(report["risk_assessment"]["text"])
    groups = st.tabs(["Risk indicators", "Feature evidence", "Historical context", "Review actions"])
    with groups[0]:
        for title, key in (("Risk-increasing", "suspicious_indicators"), ("Risk-reducing", "reassuring_indicators")):
            st.markdown(f"**{title}**")
            for item in report[key]:
                st.text(item["text"])
    with groups[1]:
        for item in report["feature_evidence"]:
            st.text(f"{item['text']} · {item['evidence_id']}")
    with groups[2]:
        st.text(report["historical_context"]["text"])
    with groups[3]:
        for item in report["recommended_review_actions"]:
            st.text(item["text"])
    st.caption(report["limitations"]["text"])


def copilot_panel(result, engine, provider):
    st.subheader("FraudShield Copilot")
    st.caption(f"{provider.title()} · {engine.availability(provider)}")
    key = (result["transaction_id"], result["model_version"], result.get("analyzed_at"), result["risk"])
    if st.session_state.get("copilot_context") != key:
        st.session_state.copilot_context = key
        st.session_state.copilot_history = []
    if st.button("Clear conversation"):
        st.session_state.copilot_history = []
    history = st.session_state.copilot_history
    question = None
    with st.expander("Suggested questions"):
        for item in QUESTIONS:
            if st.button(item, key=f"ask_{item}"):
                question = item
    question = st.chat_input("Ask about the selected transaction", max_chars=350) or question
    if question:
        with st.spinner("Checking the available evidence…"):
            answer, source = copilot(engine, result, question, provider)
        history = [*history, (question, answer, source)][-8:]
        st.session_state.copilot_history = history
    for question, answer, source in history:
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            st.html(f'<div style="white-space:pre-wrap;overflow-wrap:anywhere">{escape(answer)}</div>')
            st.caption(f"{source.title()} · Selected transaction evidence")
