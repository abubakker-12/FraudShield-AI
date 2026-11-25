"""Shared transaction catalog, scoring, and investigation workflow."""
import json
import os
from datetime import datetime, timezone
from functools import cached_property, lru_cache
from threading import Lock
from time import perf_counter

import joblib
import pandas as pd

from .explanations import ExplanationEngine
from .features import CAT, FIELDS, RAW, ROOT, engineer_features, features
from .storage import Store
from .presentation import display
from .reports import LOCAL_FIELDS, report


class ModelUnavailable(RuntimeError):
    """The trusted packaged classifier could not be loaded."""


def selected_transaction(page, positions):
    """Resolve the visible row selection against the same page sent to the UI."""
    if not positions or not 0 <= positions[0] < len(page):
        return None
    return str(page.iloc[positions[0]].transaction_id)


class Catalog:
    def __init__(self, path):
        self.path = path
        self.synthetic = path != RAW

    @cached_property
    def data(self):
        columns = ["transaction_id", *FIELDS] + ([] if self.synthetic else ["is_fraud", "fraud_type", *LOCAL_FIELDS])
        data = pd.read_csv(self.path, usecols=columns).set_index("transaction_id")
        if data.index.hasnans or not data.index.is_unique:
            raise ValueError("Transaction IDs must be present and unique")
        return data

    def browse(self, query="", international=None, payment=None, minimum=0, offset=0, limit=50,
               *, maximum=None, merchant=None, device=None, failures=None, sort=None):
        df = self.data
        mask = df.transaction_amount.ge(minimum)
        if query:
            mask &= df.index.str.contains(query, case=False, regex=False)
        if international is not None:
            mask &= df.is_international.eq(international)
        if payment:
            mask &= df.payment_method.eq(payment)
        if maximum is not None:
            mask &= df.transaction_amount.le(maximum)
        if merchant:
            mask &= df.merchant_category.eq(merchant)
        if device:
            mask &= df.device_type.eq(device)
        if failures is not None:
            mask &= df.failed_attempts.ge(failures)
        ids = df.index[mask]
        if limit == 0:
            return len(ids), df.iloc[:0][FIELDS].reset_index()
        if sort in {"amount_high", "amount_low", "failures"}:
            column = "failed_attempts" if sort == "failures" else "transaction_amount"
            ids = df.loc[ids, column].sort_values(ascending=sort == "amount_low", kind="stable").index
        page = df.loc[ids[offset:offset + limit], FIELDS].reset_index()
        return len(ids), page

    def get(self, transaction_id):
        return self.data.loc[transaction_id, FIELDS].to_dict()

    def copilot_attributes(self, transaction_id):
        if self.synthetic or transaction_id not in self.data.index:
            return {}
        row = self.data.loc[transaction_id, list(LOCAL_FIELDS)]
        return row.dropna().to_dict()

    def actual(self, transaction_id):
        if self.synthetic:
            return None
        row = self.data.loc[transaction_id]
        return {"is_fraud": int(row.is_fraud), "fraud_type": None if pd.isna(row.fraud_type) else row.fraud_type}


class Scorer:
    def __init__(self):
        directory = ROOT / "models"
        try:
            self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            # Only deserialize the trusted, locally shipped model artifact.
            artifact = joblib.load(directory / "fraudshield.joblib")
            self.model = artifact["model"]
            self.prep = artifact["preprocessor"]
            self.background = artifact["background"]
            self.names = self.prep.get_feature_names_out().tolist()
        except (OSError, ValueError, EOFError, KeyError, TypeError, AttributeError, ImportError) as error:
            raise ModelUnavailable("Trusted model artifacts could not be loaded") from error
        self.lock = Lock()
        self.explainer = None

    def _score(self, matrix):
        return self.model.predict_proba(matrix)[:, 1]

    def result(self, score):
        cut = self.metadata["threshold"]
        return dict(risk=float(score), threshold=cut, suspected_fraud=bool(score >= cut),
                    model_version=self.metadata["version"])

    def batch(self, frame):
        matrix = self.prep.transform(features(frame))
        return [self.result(score) for score in self._score(matrix)]

    def score_validated_batch(self, frame):
        return self._score(self.prep.transform(engineer_features(frame)))

    def explain(self, row):
        values = tuple(row[name] for name in FIELDS)
        risk, base, inference_ms, attributions = self._explain_values(values)
        result = self.result(risk)
        result["base_score"] = base
        result["inference_ms"] = inference_ms
        result["attributions"] = [dict(feature=name, contribution=value, feature_value=raw)
                                  for name, value, raw in attributions]
        return result

    @lru_cache(maxsize=256)
    def _explain_values(self, values):
        start = perf_counter()
        prepared = features(pd.DataFrame([values], columns=FIELDS))
        matrix = self.prep.transform(prepared)
        risk = self._score(matrix)[0]
        inference_ms = (perf_counter()-start)*1000
        with self.lock:
            if self.explainer is None:
                import shap
                self.explainer = shap.PermutationExplainer(self._score, self.background, seed=42)
            item = self.explainer(matrix, max_evals=4 * len(self.names) + 2, silent=True)[0]
        contributions = dict(zip(self.names, item.values.tolist()))
        ordered = tuple(sorted(contributions.items(), key=lambda pair: abs(pair[1]), reverse=True))
        raw = prepared.iloc[0]

        def real_value(name):
            if name in raw.index:
                return display(name, raw[name]) if name in FIELDS else f"{raw[name]:.4g}"
            field = next(field for field in CAT if name.startswith(field + "_"))
            return "Yes" if raw[field] == name[len(field)+1:] else "No"

        return float(risk), float(item.base_values), inference_ms, tuple(
            (name, value, real_value(name)) for name, value in ordered)


class InvestigationService:
    def __init__(self, catalog, scorer, store, engine):
        self.catalog, self.scorer, self.store, self.engine = catalog, scorer, store, engine

    def analyze(self, transaction_id, provider=None):
        return self.analyze_row(transaction_id, self.catalog.get(transaction_id), provider)

    def analyze_row(self, transaction_id, row, provider=None):
        start = perf_counter()
        result = self.scorer.explain(row)
        safe_row = dict(row)
        for name in CAT:
            if safe_row[name] not in self.scorer.metadata["categories"][name]:
                safe_row[name] = "Unknown category"
        result["explanation"] = report(self.engine, result, safe_row,
                                      self.scorer.metadata["training_evidence"], provider)
        result["transaction"] = {name: row[name] for name in FIELDS}
        result["analyzed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result["analysis_ms"] = round((perf_counter()-start)*1000, 2)
        self.store.save(transaction_id, result)
        return {"transaction_id": transaction_id, **result}


@lru_cache(maxsize=1)
def defaults():
    path = RAW if os.getenv("FRAUDSHIELD_USE_LOCAL_DATA") == "1" and RAW.exists() else ROOT / "data" / "demo.csv"
    database = ROOT / os.getenv("DATABASE_PATH", "database/fraudshield.db")
    return InvestigationService(Catalog(path), Scorer(), Store(database), ExplanationEngine())
