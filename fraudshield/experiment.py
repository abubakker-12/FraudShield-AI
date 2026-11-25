"""Offline CPU model comparison that reserves the original holdout."""
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .features import CAT, ENGINEERED, NUM, ROOT, features
from .eda import read_source, groups
from .service import Scorer
from .train import evaluate, split


def run():
    df, stamp = read_source()
    (train, val, test), split_info = split(df, stamp)
    test_count = len(test)
    del df, test
    # Earlier customer buckets within the original validation period calibrate; later
    # disjoint buckets rank models and pick operating points. The original test is untouched.
    boundary = stamp.loc[val.index].quantile(.4)
    bucket = pd.util.hash_pandas_object(val.customer_id, index=False) % 101
    calibration = val.loc[(stamp.loc[val.index] < boundary) & (bucket < 45)]
    validation = val.loc[(stamp.loc[val.index] >= boundary) & (bucket >= 45)]
    sampled = train.sample(min(220_000, len(train)), random_state=42)
    base = Scorer()

    def engineered(df):
        x = features(df)
        x["merchant_period"] = x.merchant_category + "/" + x.overnight.astype(str)
        return x

    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUM + ENGINEERED),
        ("category", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
         CAT + ["merchant_period"]),
    ])
    x = prep.fit_transform(engineered(sampled))
    xc = prep.transform(engineered(calibration))
    xv = prep.transform(engineered(validation))
    y, yc, yv = (part.is_fraud.to_numpy() for part in (sampled, calibration, validation))
    weights = np.where(y, 4., 1.)
    candidates = {
        "Dummy": (DummyClassifier(strategy="prior"), False),
        "Logistic Regression": (LogisticRegression(max_iter=150), True),
        "Random Forest": (RandomForestClassifier(n_estimators=40, max_depth=9, min_samples_leaf=35,
                                                n_jobs=2, random_state=42), True),
        "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                              early_stopping=False, random_state=42), True),
        **{f"XGBoost d{depth} w{weight}": (XGBClassifier(max_depth=depth, n_estimators=240,
            learning_rate=.06, early_stopping_rounds=18, scale_pos_weight=weight,
            tree_method="hist", n_jobs=2, random_state=42), False)
            for depth, weight in [(2, 1), (3, 4), (4, 8)]},
    }
    skipped = {}
    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = (LGBMClassifier(n_estimators=120, num_leaves=15, n_jobs=2,
                                                verbosity=-1, random_state=42), True)
    except ImportError:
        skipped["LightGBM"] = "Offline dependency unavailable"
    try:
        from catboost import CatBoostClassifier
        candidates["CatBoost"] = (CatBoostClassifier(iterations=120, depth=4, thread_count=2,
            verbose=False, allow_writing_files=False, random_seed=42), True)
    except ImportError:
        skipped["CatBoost"] = "Offline dependency unavailable"

    def metrics(score, cut, fit=0., infer=0.):
        result = evaluate(yv, score, cut, infer, fit)
        p, r = result["precision"], result["recall"]
        return {**result, "f2": 5*p*r/max(4*p+r, 1e-15), "brier": brier_score_loss(yv, score)}

    def strategies(score):
        p, r, cuts = precision_recall_curve(yv, score)
        p, r = p[:-1], r[:-1]
        fp = r * yv.sum() * (1 / np.maximum(p, 1e-15) - 1)
        choices = {"balanced": np.argmax(2*p*r / np.maximum(p+r, 1e-15)),
                   "recall_oriented": np.argmax(5*p*r / np.maximum(4*p+r, 1e-15)),
                   "precision_oriented": np.argmax(np.where(r >= .10, p, -1)),
                   "cost_sensitive": np.argmin(fp + 10 * yv.sum() * (1-r))}
        return {name: metrics(score, float(cuts[index])) for name, index in choices.items()}

    base_matrix = base.prep.transform(features(validation))
    start = perf_counter()
    base_score = base.model.predict_proba(base_matrix)[:, 1]
    base_infer = perf_counter() - start
    base_fit = base.metadata["validation"][base.metadata["model"]]["fit_seconds"]
    baseline = metrics(base_score, base.metadata["threshold"], base_fit, base_infer)
    base_cal_score = np.array([entry["risk"] for entry in base.batch(calibration)])
    base_cal = LogisticRegression().fit(base_cal_score[:, None], yc)
    calibrated_base_options = strategies(base_cal.predict_proba(base_score[:, None])[:, 1])
    results = {}
    for name, (model, weighted) in candidates.items():
        start = perf_counter()
        args = {"sample_weight": weights} if weighted else {}
        if isinstance(model, XGBClassifier):
            # Internal earlier training slice for early stopping, disjoint from calibration.
            early = train.drop(index=sampled.index).sample(8500, random_state=4)
            args["eval_set"] = [(prep.transform(engineered(early)), early.is_fraud)]
            args["verbose"] = False
        model.fit(x, y, **args)
        fit = perf_counter() - start
        calibrator = LogisticRegression().fit(model.predict_proba(xc)[:, 1, None], yc)
        start = perf_counter()
        raw = model.predict_proba(xv)[:, 1]
        calibrated = calibrator.predict_proba(raw[:, None])[:, 1]
        infer = perf_counter() - start
        options = strategies(calibrated)
        balanced = metrics(calibrated, options["balanced"]["threshold"], fit, infer)
        results[name] = {"balanced": balanced, "strategies": options,
                         "uncalibrated_brier": brier_score_loss(yv, raw)}
        print(name, round(balanced["average_precision"], 6), flush=True)
    best = max(results, key=lambda key: results[key]["balanced"]["average_precision"])
    improvement = results[best]["balanced"]["average_precision"] - baseline["average_precision"]
    # Small one-window ranking swings are insufficient evidence for an artifact change.
    report = {"fit_rows": len(y), "calibration_rows": len(yc), "validation_rows": len(yv),
              "test_rows_preserved": test_count, "test_rescored": False,
              "split": split_info, "prevalence": { "fit": float(y.mean()), "selection": float(yv.mean())},
              "baseline": baseline, "calibrated_baseline_diagnostic": calibrated_base_options,
              "candidates": results, "skipped": skipped,
              "best_candidate": best, "ap_change": improvement, "promotion_minimum_ap_gain": .015,
              "active_model": base.metadata["version"],
              "decision": "Retain baseline" if improvement < .015 else "Requires promotion evaluation",
              "cost_assumptions": "One unit per false positive, ten per false negative, zero for correct classifications; illustrative only.",
              "feature_audit": {"balance_and_frequency": "Excluded: snapshot timing and historical window provenance unknown",
                                "merchant_timing": "Joint category evaluated alongside baseline inputs",
                                "individual_numeric_ap": {} }}
    from sklearn.metrics import average_precision_score
    values = features(validation)
    report["feature_audit"]["individual_numeric_ap"] = {
        col: max(average_precision_score(yv, values[col]), average_precision_score(yv, -values[col]))
        for col in NUM + ENGINEERED}
    (ROOT / "reports" / "experiment-v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(best, improvement, report["decision"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
