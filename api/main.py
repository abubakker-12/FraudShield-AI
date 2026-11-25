import logging
import os
import sqlite3
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fraudshield.service import InvestigationService, ModelUnavailable, defaults
from fraudshield.storage import ReviewStatus
from fraudshield.reports import InvestigationReport

app = FastAPI(title="FraudShield AI", version="1.0")
logger = logging.getLogger(__name__)
def local_api_only():
    if os.getenv("FRAUDSHIELD_PUBLIC_DEMO") == "1":
        raise HTTPException(403, "Use the session-isolated Streamlit demo")


app.router.dependencies.append(Depends(local_api_only))
Service = Annotated[InvestigationService, Depends(defaults)]
ShortText = Annotated[str, Field(min_length=1, max_length=80)]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hour_of_day: int = Field(ge=0, le=23)
    is_international: int = Field(ge=0, le=1)
    failed_attempts: int = Field(ge=0, le=1000)
    pin_changed_recently: int = Field(ge=0, le=1)
    time_since_last_txn_hrs: float = Field(ge=0, le=1e6)
    transaction_amount: float = Field(ge=.01, le=1e12)
    merchant_category: ShortText
    payment_method: ShortText
    device_type: ShortText


class CatalogEntry(Transaction):
    transaction_id: ShortText


class CatalogPage(BaseModel):
    total: int
    transactions: list[CatalogEntry]


class Score(BaseModel):
    risk: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    suspected_fraud: bool
    model_version: str


class Attribution(BaseModel):
    feature: str
    contribution: float
    feature_value: str | None = None


class Narrative(BaseModel):
    provider: str
    assessment: str
    increasing: list[Attribution]
    decreasing: list[Attribution]
    reasoning: str
    actions: list[str]
    context: str | None
    provider_model: str | None = None
    report: InvestigationReport | None = None
    limitations: str


class Analysis(Score):
    transaction_id: str
    base_score: float
    attributions: list[Attribution]
    explanation: Narrative
    transaction: Transaction | None = None
    analyzed_at: str | None = None
    inference_ms: float | None = None
    analysis_ms: float | None = None


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: ShortText


class Review(BaseModel):
    status: ReviewStatus
    notes: str | None = Field(None, max_length=1500)


class Case(BaseModel):
    id: int
    transaction_id: str
    model_version: str
    risk: float
    threshold: float
    payload: str
    provider: str
    analyzed_at: str
    status: ReviewStatus
    updated_at: str
    notes: str | None = None


@app.exception_handler(ValueError)
async def invalid_input(_request, _exc):
    logger.info("Input rejected by service validation")
    return JSONResponse(status_code=422, content={"detail": "Invalid transaction features"})


@app.exception_handler(sqlite3.Error)
async def database_error(_request, _exc):
    logger.warning("SQLite operation unavailable")
    return JSONResponse(status_code=503, content={"detail": "Investigation database unavailable"})


@app.exception_handler(ModelUnavailable)
async def model_error(_request, _exc):
    return JSONResponse(status_code=503, content={"detail": "Model artifacts unavailable"})


@app.get("/health")
def health(service: Service) -> dict[str, str]:
    return {"status": "ok", "model_version": service.scorer.metadata["version"]}


@app.get("/model-info")
def model_info(service: Service) -> dict:
    return service.scorer.metadata


@app.get("/transactions", response_model=CatalogPage)
def transactions(service: Service, q: str = Query("", max_length=80),
                 international: bool | None = None, payment: str | None = Query(None, max_length=80),
                 minimum: float = Query(0, ge=0), offset: int = Query(0, ge=0),
                 limit: int = Query(50, ge=1, le=100)):
    count, page = service.catalog.browse(q, international, payment, minimum, offset, limit)
    return {"total": count, "transactions": page.to_dict("records")}


@app.get("/transactions/{transaction_id}", response_model=CatalogEntry)
def transaction(transaction_id: ShortText, service: Service):
    try:
        return {"transaction_id": transaction_id, **service.catalog.get(transaction_id)}
    except KeyError:
        raise HTTPException(404, "Transaction not found") from None


@app.post("/predict", response_model=Score)
def predict(transaction: Transaction, service: Service):
    return service.scorer.batch(pd.DataFrame([transaction.model_dump()]))[0]


@app.post("/explain", response_model=Analysis)
def explain(selection: Selection, service: Service):
    try:
        return service.analyze(selection.transaction_id)
    except KeyError:
        raise HTTPException(404, "Transaction not found") from None


@app.post("/predict-batch", response_model=list[Score])
def predict_batch(batch: Annotated[list[Transaction], Field(min_length=1, max_length=1000)], service: Service):
    return service.scorer.batch(pd.DataFrame([item.model_dump() for item in batch]))


@app.get("/investigations", response_model=list[Case])
def investigations(service: Service, q: str = Query("", max_length=80),
                   status: ReviewStatus | None = None, offset: int = Query(0, ge=0),
                   limit: int = Query(50, ge=1, le=100)):
    return service.store.records(q, status, limit, offset)


@app.patch("/investigations/{case_id}", response_model=Review)
def review(case_id: int, review: Review, service: Service):
    if not service.store.review(case_id, review.status, review.notes):
        raise HTTPException(404, "Investigation not found")
    return review
