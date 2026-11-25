import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from api.main import app
from fraudshield.explanations import ExplanationEngine, NarrativeChoice
from fraudshield.features import FIELDS, ROOT, features
from fraudshield.service import Catalog, InvestigationService, Scorer, defaults, selected_transaction
from fraudshield.storage import Store
from fraudshield.train import split
from fraudshield.reports import ReportPlan
from fraudshield.dashboard import NAVIGATION


@pytest.fixture(scope="session")
def catalog():
    return Catalog(ROOT / "data" / "demo.csv")


@pytest.fixture(scope="session")
def scorer():
    return Scorer()


@pytest.fixture
def service(catalog, scorer, tmp_path):
    return InvestigationService(catalog, scorer, Store(tmp_path / "test.db"), ExplanationEngine())


def test_preprocessing_excludes_labels_and_identifiers(catalog, scorer):
    _, rows = catalog.browse(limit=5)
    inputs = features(rows.assign(fraud_type="Leak", is_fraud=1, customer_id="private"))
    assert set(FIELDS).issubset(inputs.columns)
    assert not {"fraud_type", "is_fraud", "transaction_id", "customer_id"} & set(inputs)
    unknown = rows.assign(payment_method="Unseen category")
    assert len(scorer.batch(unknown)) == len(rows)


@pytest.mark.parametrize("column,value", [
    ("transaction_amount", -1), ("is_international", 2), ("hour_of_day", 4.7),
    ("failed_attempts", np.inf), ("device_type", None), ("time_since_last_txn_hrs", "invalid"),
])
def test_feature_validation(catalog, column, value):
    row = catalog.browse(limit=1)[1]
    with pytest.raises(ValueError, match=column):
        features(row.assign(**{column: value}))


def test_split_customer_and_time_isolation():
    n = 9000
    df = pd.DataFrame({"customer_id": np.tile(np.arange(300), 30), "transaction_id": np.arange(n),
                       "is_fraud": np.arange(n) % 7 == 0})
    stamp = pd.Series(pd.date_range("2023-01-01", periods=n, freq="h"))
    partitions, _ = split(df, stamp)
    for a, b in zip(partitions, partitions[1:]):
        assert set(a.customer_id).isdisjoint(b.customer_id)
        assert stamp.loc[a.index].max() < stamp.loc[b.index].min()
    assert set(partitions[0].customer_id).isdisjoint(partitions[2].customer_id)


@pytest.fixture(scope="session")
def examples(catalog, scorer):
    _, rows = catalog.browse(limit=500)
    scores = scorer.batch(rows)
    return {flag: rows.iloc[next(i for i, result in enumerate(scores)
                               if result["suspected_fraud"] == flag)].transaction_id for flag in [False, True]}


@pytest.mark.parametrize("flag", [False, True])
def test_prediction_shap_and_fallback(service, examples, flag):
    result = service.analyze(examples[flag], "python")
    assert result["suspected_fraud"] is flag
    assert result["risk"] == pytest.approx(
        result["base_score"] + sum(item["contribution"] for item in result["attributions"]), abs=1e-6)
    assert result["explanation"]["provider"] == "python"
    assert "flagged" in result["explanation"]["assessment"] if flag else "below" in result["explanation"]["assessment"]
    assert all(x["contribution"] > 0 for x in result["explanation"]["increasing"])
    assert all(x["contribution"] < 0 for x in result["explanation"]["decreasing"])
    assert len(service.store.records(history=True)) == 1
    assert len(service.store.records()) == int(flag)


def test_no_duplicate_case_and_saved_review(service, examples):
    transaction_id = examples[True]
    service.analyze(transaction_id, "python")
    row = service.store.records()[0]
    assert service.store.review(row["id"], "Dismissed")
    service.analyze(transaction_id, "python")
    reopened = Store(service.store.path)
    assert len(reopened.records(history=True)) == 2
    assert len(reopened.records()) == 1
    assert reopened.records(status="Dismissed")[0]["id"] == row["id"]


@pytest.mark.parametrize("message", [
    '{"reasoning":3,"action":0}', '{"reasoning":0,"action":1,"risk":1}',
    '{"reasoning":"0","action":1}', '{"action":1}', "free text",
])
def test_reject_invalid_llm_json(message):
    with pytest.raises(ValidationError):
        NarrativeChoice.model_validate_json(message)


@pytest.mark.parametrize("failure", [requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError])
def test_outage_falls_back(monkeypatch, failure):
    engine = ExplanationEngine({"GROQ_API_KEY": "test-token", "GROQ_MODEL": "test-model"})

    def unavailable(*_args):
        raise failure()

    monkeypatch.setattr(engine, "_request", unavailable)
    prediction = {"suspected_fraud": True, "attributions": [], "risk": .9, "threshold": .5}
    assert engine.explain(prediction, {}, [], "groq")["provider"] == "python"


@pytest.mark.parametrize("model", ["test-model", "openai/gpt-oss-20b"])
def test_provider_verification_privacy_cache_and_limit(service, examples, monkeypatch, model):
    engine = ExplanationEngine({"GROQ_API_KEY": "test-token", "GROQ_MODEL": model,
                                "LLM_REQUESTS_PER_MINUTE": "1"})
    service.engine = engine
    calls = []

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response({"choices": [{"message": {"content": json.dumps(dict.fromkeys(ReportPlan.model_fields, 0))}}]})

    monkeypatch.setattr(requests, "get", lambda *a, **k: Response({"data": [{"id": model}]}))
    monkeypatch.setattr(requests, "post", post)
    tid = examples[True]
    result = service.analyze(tid, "groq")
    assert result["explanation"]["provider"] == "groq"
    assert tid not in str(calls) and "customer_id" not in str(calls) and "fraud_type" not in str(calls)
    assert service.analyze(tid, "groq")["explanation"]["provider"] == "groq"
    assert service.analyze(examples[False], "groq")["explanation"]["provider"] == "python"
    assert len(calls) == 1
    response_format = calls[0]["response_format"]
    if model == "openai/gpt-oss-20b":
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["properties"]["executive_summary"]["enum"] == [0, 1]
        assert calls[0]["reasoning_effort"] == "low"
    else:
        assert response_format == {"type": "json_object"}


def test_api_and_selected_row_workflow(service, examples, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "python")
    app.dependency_overrides[defaults] = lambda: service
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "ok"
            assert client.get("/model-info").json()["features"] == FIELDS
            assert client.get("/docs").status_code == 200
            tid = examples[True]
            result = client.get("/transactions", params={"q": tid, "limit": 1}).json()
            page = pd.DataFrame(result["transactions"])
            assert result["total"] == 1 and selected_transaction(page, [0]) == tid
            assert selected_transaction(page, []) is None
            assert selected_transaction(page, [1]) is None
            row = client.get(f"/transactions/{tid}").json()
            assert "is_fraud" not in row and "fraud_type" not in row
            assert client.get("/transactions/missing").status_code == 404
            payload = {name: row[name] for name in FIELDS}
            assert client.post("/predict", json=payload).json()["suspected_fraud"]
            assert client.post("/predict", json={**payload, "is_fraud": 1}).status_code == 422
            assert len(client.post("/predict-batch", json=[payload, payload]).json()) == 2
            assert client.post("/predict-batch", json=[]).status_code == 422
            assert client.post("/explain", json={"transaction_id": tid}).json()["explanation"]["provider"] == "python"
            case = client.get("/investigations").json()[0]
            assert client.patch(f"/investigations/{case['id']}", json={"status": "Dismissed"}).status_code == 200
            assert client.get("/investigations", params={"status": "Dismissed"}).json()[0]["id"] == case["id"]
    finally:
        app.dependency_overrides.clear()


def test_streamlit_all_pages(monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_PUBLIC_DEMO", "1")
    ui = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    assert not ui.exception
    for page in NAVIGATION:
        ui.sidebar.radio[0].set_value(page).run()
        assert not ui.exception, page
        assert ui.header[0].value == page
    ui.sidebar.radio[0].set_value("Transaction Explorer").run()
    ui.text_input[0].set_value("DEMO-0001").run()
    assert not ui.exception
    assert len(ui.dataframe[0].value) == 1


@pytest.mark.parametrize("flag", [False, True])
def test_streamlit_investigation_and_history(monkeypatch, examples, flag):
    monkeypatch.setenv("FRAUDSHIELD_PUBLIC_DEMO", "1")
    monkeypatch.setenv("LLM_PROVIDER", "python")
    original = st.dataframe

    def select_first_row(*args, **kwargs):
        rendered = original(*args, **kwargs)
        if kwargs.get("on_select") == "rerun":
            return SimpleNamespace(selection=SimpleNamespace(rows=[0]))
        return rendered

    # AppTest exposes dataframe values but does not simulate browser selection events.
    monkeypatch.setattr(st, "dataframe", select_first_row)
    ui = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    ui.sidebar.radio[0].set_value("Transaction Explorer").run()
    ui.text_input[0].set_value(examples[flag]).run()
    next(button for button in ui.button if button.label == "Analyze Transaction").click().run()
    assert not ui.exception
    assert ui.header[0].value == "Fraud Investigation"
    assert ui.session_state.analysis["transaction_id"] == examples[flag]
    assert ui.session_state.analysis["suspected_fraud"] == flag
    assert ui.error if flag else ui.success
    ui.sidebar.radio[0].set_value("Saved Fraud Cases").run()
    assert not ui.exception
    frames = [table.value for table in ui.dataframe]
    assert len(frames) == 1 + int(flag)
    assert frames[-1].iloc[0].transaction_id == examples[flag]


def test_streamlit_batch_csv(monkeypatch, catalog):
    sample = catalog.browse(limit=4)[1][FIELDS].to_csv(index=False).encode()

    def upload(*args, **kwargs):
        stream = BytesIO(sample)
        stream.size, stream.file_id = len(sample), "test-upload"
        return stream

    monkeypatch.setenv("FRAUDSHIELD_PUBLIC_DEMO", "1")
    monkeypatch.setattr(st, "file_uploader", upload)
    ui = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
    ui.sidebar.radio[0].set_value("Batch Prediction").run()
    ui.button[0].click().run()
    assert not ui.exception
    assert len(ui.dataframe[-1].value) == 4


def test_ollama_validates_model_and_response(monkeypatch):
    engine = ExplanationEngine()

    def response(content):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: content)

    def post(url, **kwargs):
        assert url.endswith("/api/chat")
        assert kwargs["json"]["model"] == "qwen3:4b"
        assert kwargs["json"]["think"] is False
        return response({"message": {"content": '{"reasoning":1,"action":0}'}})

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(requests, "get", lambda *a, **k: response({"models": [{"name": "qwen3:4b"}]}))
    monkeypatch.setattr(requests, "post", post)
    prediction = {"suspected_fraud": False, "attributions": [], "risk": .2, "threshold": .5}
    assert engine.explain(prediction, {}, [], "ollama")["provider"] == "ollama"
