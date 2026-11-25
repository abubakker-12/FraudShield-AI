"""Evidence-linked investigation reports and transaction-scoped Copilot answers."""
import logging
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .explanations import NarrativeChoice
from .presentation import display, label

logger = logging.getLogger(__name__)

Index = Annotated[int, Field(ge=0, le=2)]
QUESTIONS = {
    "Why did the model assign this score?": "decision",
    "Which factors contributed most?": "factors",
    "What should I verify?": "actions",
    "How does this compare with historical patterns?": "history",
    "Explain SHAP in simple language.": "shap",
    "What are the limitations?": "limitations",
}


class ReportPlan(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    executive_summary: Index
    risk_assessment: Index
    suspicious_indicators: Index
    reassuring_indicators: Index
    feature_evidence: Index
    historical_context: Index
    recommended_review_actions: Index
    limitations: Index


class EvidenceStatement(BaseModel):
    evidence_id: str
    text: str


class InvestigationReport(BaseModel):
    executive_summary: EvidenceStatement
    risk_assessment: EvidenceStatement
    suspicious_indicators: list[EvidenceStatement]
    reassuring_indicators: list[EvidenceStatement]
    feature_evidence: list[EvidenceStatement]
    historical_context: EvidenceStatement
    recommended_review_actions: list[EvidenceStatement]
    limitations: EvidenceStatement


def statement(ref, text):
    return {"evidence_id": ref, "text": text}


def report(engine, prediction, characteristics, findings, provider=None):
    envelope = engine.explain(prediction, characteristics, findings, "python")
    risk, threshold = prediction["risk"], prediction["threshold"]
    flagged = prediction["suspected_fraud"]
    decision = "Suspected fraud" if flagged else "Likely legitimate"
    determination = f"{decision}. The model score was {risk:.2%} against a {threshold:.2%} review threshold."
    base = prediction["base_score"]
    empty = statement("shap", "No nonzero contribution of this direction was available.")
    factors = {"up": [], "down": []}
    for index, item in enumerate(prediction["attributions"]):
        amount = item["contribution"]
        if abs(amount) <= 1e-9:
            continue
        direction = "up" if amount > 0 else "down"
        if len(factors[direction]) < 3:
            value = item.get("feature_value", "unavailable")
            factors[direction].append(statement(f"shap:{index}",
                f"{label(item['feature'])} ({value}) changed the score by {amount*100:+.2f} percentage points."))
    up, down = factors["up"] or [empty], factors["down"] or [empty]
    primary = up[0] if flagged else down[0]
    summary = [statement("decision", determination),
               statement(f"decision+{primary['evidence_id']}", f"{determination} {primary['text']}")]
    actions = (
        [statement("decision", "Verify authorization using established review procedures."),
         statement("transaction:failed_attempts", "Review available authentication records and verify whether retries were expected."),
         statement("transaction:merchant_category", "Check merchant and payment context against independently verified evidence.")]
        if flagged else
        [statement("decision", "Continue routine monitoring for this transaction."),
         statement("decision", "Escalate for manual review if independently verified evidence warrants it.")]
    )
    options = {
        "executive_summary": summary,
        "risk_assessment": [statement("decision", determination)],
        "suspicious_indicators": up,
        "reassuring_indicators": down,
        "feature_evidence": (up[:2] + down[:1]) if flagged else (down[:2] + up[:1]),
        "historical_context": [statement("training:repeat_failures", envelope["context"] or
                                        "A comparable verified training cohort is unavailable.")],
        "recommended_review_actions": actions,
        "limitations": [statement("model:limitations", envelope["limitations"])],
    }
    evidence = {
        "characteristics": characteristics,
        "prediction": {"classification": decision, "score": round(risk, 6),
                       "threshold": round(threshold, 6), "shap_reference": round(base, 6)},
        "report_options": options,
    }
    # Every field is required by the provider schema, even when there is one possible reference.
    choice, source = engine.choose(provider or engine.get("LLM_PROVIDER", "groq"), evidence, default_plan(), options)
    multiple = {"suspicious_indicators", "reassuring_indicators", "feature_evidence", "recommended_review_actions"}
    output = {}
    for name, index in choice.model_dump().items():
        entries = options[name]
        output[name] = ([entries[index], *entries[:index], *entries[index+1:]] if name in multiple else entries[index])
    result = InvestigationReport.model_validate(output).model_dump()
    return {**envelope, "provider": source, "provider_model": provider_model(engine, source),
            "reasoning": result["executive_summary"]["text"],
            "actions": [item["text"] for item in result["recommended_review_actions"]],
            "report": result}


def default_plan():
    return ReportPlan(**{**dict.fromkeys(ReportPlan.model_fields, 0), "executive_summary": 1})


def provider_model(engine, provider):
    return {"groq": lambda: engine.get("GROQ_MODEL"),
            "ollama": lambda: engine.get("OLLAMA_MODEL", "qwen3:4b"),
            "python": lambda: "Deterministic Python"}[provider]()


FACT_TOPICS = {
    "time_since_last_txn_hrs": r"time since|previous transaction|last transaction|how (long|recently)|\binterval\b",
    "transaction_date": r"\b(date|calendar day)\b|which day",
    "time": r"\b(time|hour|hr|minute|timestamp|timezone|when|tme|tiem)\b|what time",
    "transaction_amount": r"\b(amount|amout|ammount|cost|spent|value|sum|currency)\b|how much",
    "merchant_category": r"\b(merchant|store|retailer|category|seller)\b",
    "payment_method": r"\b(payment|pay|paid|card|transfer)\b",
    "device_type": r"\b(device|mobile|desktop|terminal|channel|devce)\b",
    "is_international": r"\b(international|domestic|cross.border|foreign|abroad)\b",
    "location": r"\b(city|country|location|geography|where)\b",
    "customer_age": r"\bcustomer age\b|how old.*(customer|person)|\bage\b(?! of (the )?account)",
    "account_age_years": r"\baccount age\b|age of (?:the )?account|how old.*account|account tenure",
    "credit_score": r"\bcredit (score|rating)\b",
    "account_balance": r"\b(balance|funds)\b",
    "failed_attempts": r"\b(fail(?:ed|ures?)|retries|retry|attempts|authentication)\b",
    "pin_changed_recently": r"\bpin\b",
    "transaction_freq_monthly": r"\b(frequency|how often|monthly transactions|per month)\b",
    "num_prev_transactions": r"\b(number of previous transactions|prior transaction count|how many previous transactions)\b",
    "distance_from_home_km": r"\b(distance|kilometers|kilometres|how far)\b",
    "is_weekend": r"\b(weekend|weekday|saturday|sunday)\b",
    "is_night_transaction": r"\b(night|overnight|late evening|daytime)\b",
    "review_status": r"\b(case status|review status|human decision|dismissed|pending|confirmed|under investigation)\b",
    "decision_fact": r"\b(score|threshold|cutoff|prediction|classification|flagged|legitimate|suspected fraud|risky)\b",
}
LOCAL_FIELDS = (
    "transaction_date", "transaction_time", "country", "city", "customer_age", "account_age_years",
    "credit_score", "account_balance", "num_prev_transactions", "transaction_freq_monthly",
    "distance_from_home_km", "is_weekend", "is_night_transaction",
)
ANALYTICAL = {
    "limitations": r"limitation|uncertain|confidence|calibrat",
    "amount_comparison": r"unusual|expensive|typical|large amount|high amount|above average",
    "history": r"historical|cohort|pattern|past fraud",
    "actions": r"verify|review action|investigator|investigate next|what should i (?:check|do)|next step",
    "shap": r"explain shap|simple.*shap|shap.*(?:simple|work|mean)|reference (?:value|score)|reconstruct",
    "factors": r"factor|contribut|indicator|shap",
    "decision": r"\bwhy\b.*(?:flag|score|legitimate|risk|prediction|suspicio|fraud)|what made .*suspicio",
}
BLOCKED = re.compile(
    r"\b(ignore|override|prompt|instructions?|system|secrets?|passwords?|credentials?|api key|sql|execute)\b"
    r"|\b(?:other|another|unrelated|different)\s+(?:transaction|customer|user|account|case)"
    r"|\b(?:DEMO|TXN|CUST|BATCH)-?[0-9]", re.IGNORECASE,
)


def question_topics(question, previous=None):
    text = question.casefold().replace("-", " ")
    if BLOCKED.search(question):
        return []
    if question in QUESTIONS:
        return [QUESTIONS[question]]
    topics = []
    for clause in re.split(r"[?;\n]+|\b(?:and|also|plus)\b|,(?=\s*(?:what|when|why|which|how|was|did|where)\b)", text):
        clause = clause.strip()
        if not clause:
            continue
        analytical = next((topic for topic, pattern in ANALYTICAL.items() if re.search(pattern, clause)), None)
        if not analytical and previous in {"transaction_amount", "amount_comparison"} and re.search(
            r"\b(?:it|that)\b.*\b(?:high|low|large|small|average)\b", clause
        ):
            analytical = "amount_comparison"
        if analytical == "amount_comparison" and previous not in {None, "transaction_amount", "amount_comparison"}:
            if not re.search(FACT_TOPICS["transaction_amount"], clause):
                analytical = None
        if analytical:
            topics.append(analytical)
            previous = analytical
            continue
        if re.fullmatch(r"why|why is that|why so", clause) and previous in {"decision", "decision_fact", "factors"}:
            topics.append("decision")
            continue
        matches = [field for field, pattern in FACT_TOPICS.items() if re.search(pattern, clause)]
        if "num_prev_transactions" in matches:
            matches = [field for field in matches if field != "time_since_last_txn_hrs"]
        if "time_since_last_txn_hrs" in matches:
            matches = [field for field in matches if field != "time"]
        if "credit_score" in matches:
            matches = [field for field in matches if field != "decision_fact"]
        if "account_age_years" in matches:
            matches = [field for field in matches if field != "customer_age"]
        topics.extend(matches)
        if matches:
            previous = matches[-1]
        if not matches and previous and re.search(r"\b(?:that|it|again|earlier|previously|before)\b", clause):
            topics.append(previous)
    return list(dict.fromkeys(topics))[:8]


def _fact(topic, result, row, status):
    if topic == "time":
        if row.get("transaction_time"):
            return f"Recorded transaction time: {row['transaction_time']}. The source does not specify a timezone."
        if row.get("hour_of_day") is not None:
            return (f"Recorded transaction hour: {int(row['hour_of_day']):02d}:00 "
                    "(hour only, 24-hour clock). Exact minutes, seconds and timezone are unavailable.")
        return "The selected record has no recorded transaction time or hour."
    if topic == "review_status":
        return (f"Human review status: {status}." if status else
                "No human review status is recorded for this transaction and model version.")
    if topic == "decision_fact":
        finding = "Suspected fraud" if result["suspected_fraud"] else "Likely legitimate"
        return (f"Model prediction: {finding}. Uncalibrated score {result['risk']:.2%}; "
                f"review threshold {result['threshold']:.2%}. This is separate from a human review decision.")
    if topic == "location":
        parts = [f"{label(key)}: {row[key]}" for key in ("city", "country") if row.get(key)]
        if row.get("is_international") is not None:
            parts.append(f"Transaction scope: {display('is_international', row['is_international'])}")
        if not (row.get("city") and row.get("country")):
            parts.append("Exact city and country information is incomplete or unavailable.")
        return ". ".join(parts).rstrip(".") + "."
    value = row.get(topic)
    if topic == "is_night_transaction" and value is None and row.get("hour_of_day") is not None:
        hour = int(row["hour_of_day"])
        return (f"{'Yes' if hour < 7 or hour >= 22 else 'No'} under the model's 22:00–06:59 overnight convention "
                f"(recorded hour {hour:02d}:00). A source night flag is unavailable.")
    if value is None:
        return f"{label(topic)} is unavailable in the selected record."
    if topic in {"is_weekend", "is_night_transaction"}:
        value = "Yes" if value else "No"
    else:
        value = display(topic, value)
    suffix = " Currency is not recorded." if topic in {"transaction_amount", "account_balance"} else ""
    return f"{label(topic)}: {value}.{suffix}"


def copilot(engine, result, question, provider, *, history=(), attributes=None, review_status=None, amount_stats=None):
    previous = None
    for earlier, _, _ in history[-8:]:
        earlier_topics = question_topics(earlier, previous)
        if earlier_topics:
            previous = earlier_topics[-1]
    topics = question_topics(question, previous)
    if not topics:
        logger.info("Copilot route unavailable")
        return ("Sufficient evidence is unavailable for that question. Ask about the selected transaction’s "
                "recorded details, score, SHAP evidence, historical cohort, or review actions."), "python"
    report = result.get("explanation", {}).get("report") or {}
    row = {**result.get("transaction", {}), **(attributes or {})}
    answers = []
    analytical = []
    for topic in topics:
        logger.info("Copilot route: %s", topic)
        if topic in FACT_TOPICS:
            answers.append(_fact(topic, result, row, review_status))
        elif topic == "amount_comparison":
            value = row.get("transaction_amount")
            cutoff = (amount_stats or {}).get("75%")
            if value is None or cutoff is None:
                answers.append("A recorded amount or verified historical amount benchmark is unavailable.")
            else:
                relation = "above" if value > cutoff else "at or below"
                answers.append(f"The recorded amount {value:,.2f} was {relation} the historical 75th percentile "
                               f"of {cutoff:,.2f} [historical dataset]. Currency and customer-specific spending history are unavailable.")
        else:
            analytical.append(topic)

    groups = {
        "decision": [report.get("risk_assessment"), report.get("executive_summary")],
        "factors": sorted(report.get("suspicious_indicators", []) + report.get("reassuring_indicators", []),
                          key=lambda item: int(item["evidence_id"].split(":")[-1])
                          if re.fullmatch(r"shap:\d+", item["evidence_id"]) else 9999),
        "actions": report.get("recommended_review_actions", []),
        "history": [report.get("historical_context")],
        "shap": [statement("shap", f"The score starts from a SHAP reference of {result.get('base_score', 0):.2%}. "
                           "Positive changes raise it and negative changes lower it. Together they reconstruct the model score.")
                 if result.get("base_score") is not None else None, *report.get("feature_evidence", [])[:1]],
        "limitations": [report.get("limitations")],
    }
    facts = [item for topic in analytical for item in groups[topic] if item]
    for topic in analytical:
        if not any(groups[topic]):
            answers.append(f"{topic.replace('_', ' ').capitalize()} evidence is unavailable in the selected investigation.")
    if ("factors" in analytical and re.search(r"\b(?:most|largest|strongest)\b", question, re.IGNORECASE)
            and groups["factors"] and groups["factors"][0]["evidence_id"] != "shap"):
        strongest = groups["factors"][0]
        answers.append(f"Largest absolute SHAP contribution: {strongest['text']} [{strongest['evidence_id']}]")
    source = "python"
    if facts:
        # Only report statements already built from allowlisted evidence can leave the session.
        # Raw questions, local attributes, notes and history are never sent.
        options = [item["text"] for item in facts[:2]]
        if len(options) == 1:
            options *= 2
        choice, source = engine.choose(provider, {"approved_question_intent": ",".join(analytical),
                                       "reasoning_options": options, "action_options": ["Explain", "Explain"]},
                                      NarrativeChoice(reasoning=0, action=0))
        ranked = [facts[min(choice.reasoning, len(facts)-1)], *facts]
        if answers:
            answers.append(f"Report evidence ({source.title()}):")
        answers.extend(f"{item['text']} [{item['evidence_id']}]" for item in ranked)
    if "currency" in question.casefold() and "transaction_amount" not in topics:
        answers.append("Currency is not recorded.")
    return "\n\n".join(dict.fromkeys(answers)), source
