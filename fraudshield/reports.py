"""Evidence-linked investigation reports and transaction-scoped Copilot answers."""
import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .explanations import NarrativeChoice
from .presentation import label

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


def question_intent(question):
    if question in QUESTIONS:
        return QUESTIONS[question]
    text = question.casefold()
    if re.search(r"\b(ignore|override|prompt|instruction|system|secret|password|credential|api key|sql|execute)\b"
                 r"|\b(other|another|unrelated)\s+(transaction|customer|user|account|case)", text):
        return None
    for pattern, intent in [
        (r"limitation|uncertain|confidence|calibrat", "limitations"),
        (r"histor|cohort|pattern", "history"), (r"verify|review action|investigator", "actions"),
        (r"simple|explain shap|reference value", "shap"),
        (r"factor|contribut|indicator", "factors"),
        (r"\bwhy\b.*(flag|score|legitimate|risk|prediction)", "decision"),
    ]:
        if re.search(pattern, text):
            return intent
    return None


def copilot(engine, result, question, provider):
    intent = question_intent(question)
    report = result["explanation"].get("report")
    if not intent or not report:
        return "Sufficient evidence is unavailable for that question. Ask about the selected transaction’s model score, SHAP evidence, historical cohort, or review actions.", "python"
    groups = {
        "decision": [report["risk_assessment"], report["executive_summary"]],
        "factors": report["feature_evidence"],
        "actions": report["recommended_review_actions"],
        "history": [report["historical_context"]],
        "shap": [statement("shap", f"The score starts from a SHAP reference of {result['base_score']:.2%}. "
                           "Positive changes raise it and negative changes lower it. Together they reconstruct the model score."),
                 report["feature_evidence"][0]],
        "limitations": [report["limitations"]],
    }
    facts = groups[intent]
    answers = [item["text"] for item in facts[:2]]
    if len(answers) == 1:
        answers *= 2
    choice, source = engine.choose(provider, {"approved_question_intent": intent,
                                  "reasoning_options": answers, "action_options": ["Explain", "Explain"]},
                                   NarrativeChoice(reasoning=0, action=0))
    ranked = [facts[min(choice.reasoning, len(facts)-1)], *facts]
    lines = list(dict.fromkeys(f"{item['text']} [{item['evidence_id']}]" for item in ranked))
    return "\n\n".join(lines), source
