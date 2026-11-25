"""Offline, CPU-bounded training. Run with python -m fraudshield.train."""
import hashlib
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .eda import evidence, read_source
from .features import BOUNDS, CAT, ENGINEERED, FIELDS, NUM, ROOT, features


def split(df, stamp):
    # Timestamp ties stay on one side. Stable customer buckets are independent of the target.
    first, second = stamp.quantile([.65, .82])
    buckets = pd.util.hash_pandas_object(df.customer_id, index=False) % 100
    periods = [stamp < first, (stamp >= first) & (stamp < second), stamp >= second]
    customers = [buckets < 65, (buckets >= 65) & (buckets < 82), buckets >= 82]
    if df.customer_id.isna().any() or df.transaction_id.isna().any() or df.transaction_id.duplicated().any():
        raise ValueError("Customer IDs must be present and transaction IDs unique")
    sets = [df.loc[p & c] for p, c in zip(periods, customers)]
    if any(part.is_fraud.nunique() != 2 for part in sets):
        raise ValueError("Each split requires both classes")
    return sets, {
        "boundaries": [str(first), str(second)],
        "period_customer_overlap_before_grouping": [
            len(set(df.loc[periods[i], "customer_id"]) &
                set(df.loc[periods[j], "customer_id"])) for i, j in [(0, 1), (0, 2), (1, 2)]
        ],
        "retained": [len(part) for part in sets],
        "fraud_rates": [float(part.is_fraud.mean()) for part in sets],
    }


def threshold(y, score):
    p, r, cuts = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(cuts[np.argmax(f1)])


def evaluate(y, score, cut, elapsed, fit_seconds):
    pred = score >= cut
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, score), average_precision=average_precision_score(y, score),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                false_positive_rate=float(fp / (fp + tn)), false_negative_rate=float(fn / (fn + tp)),
                fit_seconds=fit_seconds, inference_seconds=elapsed, threshold=cut)


def make_demo(train):
    # Independent draws from coarse feature categories and ranges, with fictional IDs.
    rng = np.random.default_rng(82)
    count = 500
    demo = pd.DataFrame({
        "transaction_id": [f"DEMO-{n:04d}" for n in range(1, count + 1)],
        "hour_of_day": rng.integers(0, 24, count),
        "is_international": rng.binomial(1, .23, count),
        "failed_attempts": rng.choice(6, count, p=[.60, .15, .10, .08, .05, .02]),
        "pin_changed_recently": rng.binomial(1, .08, count),
        "time_since_last_txn_hrs": np.round(rng.exponential(5, count), 2),
        "transaction_amount": np.round(rng.lognormal(4.4, 1.2, count), 2),
        **{name: rng.choice(sorted(train[name].unique()), count) for name in CAT},
    })
    (ROOT / "data").mkdir(exist_ok=True)
    demo.to_csv(ROOT / "data" / "demo.csv", index=False)
    return demo


def run():
    df, stamp = read_source()
    (train, val, test), split_info = split(df, stamp)
    del df, stamp
    train = train.sample(min(100_000, len(train)), random_state=42)
    split_info["fitted_rows"] = len(train)
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), NUM + ENGINEERED),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ], verbose_feature_names_out=False)
    x_train = prep.fit_transform(features(train))
    x_val = prep.transform(features(val))
    y_train, y_val = train.is_fraud, val.is_fraud
    weight = float(y_train.eq(0).sum() / y_train.eq(1).sum())
    candidates = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", C=.5, max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=48, max_depth=8, min_samples_leaf=25,
                                                class_weight="balanced", n_jobs=2, random_state=42),
        **{f"XGBoost depth {depth}": XGBClassifier(
            n_estimators=100, max_depth=depth, learning_rate=.08, tree_method="hist",
            scale_pos_weight=weight, n_jobs=2, random_state=42) for depth in (2, 4)},
    }
    results = {}
    for name, model in candidates.items():
        start = perf_counter()
        model.fit(x_train, y_train)
        elapsed = perf_counter() - start
        start = perf_counter()
        score = model.predict_proba(x_val)[:, 1]
        infer = perf_counter() - start
        results[name] = evaluate(y_val, score, threshold(y_val, score), infer, elapsed)
        print(name, results[name]["average_precision"], flush=True)
    winner = max(results, key=lambda name: (results[name]["average_precision"], results[name]["f1"]))
    model = candidates[winner]
    cut = results[winner]["threshold"]
    x_test = prep.transform(features(test))
    start = perf_counter()
    score = model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(test.is_fraud, score, cut, perf_counter() - start,
                            results[winner]["fit_seconds"])
    fpr, tpr, _ = roc_curve(test.is_fraud, score)
    p, r, _ = precision_recall_curve(test.is_fraud, score)
    names = prep.get_feature_names_out().tolist()
    # Ship a synthetic SHAP reference, so the model bundle contains no training rows.
    demo = make_demo(train)
    background = prep.transform(features(demo.sample(64, random_state=42)))
    # Model-agnostic permutation SHAP explains fraud score directly for every candidate type.
    explainer = shap.PermutationExplainer(lambda x: model.predict_proba(x)[:, 1], background, seed=42)
    sample = x_test[np.random.default_rng(81).choice(len(x_test), min(128, len(x_test)), replace=False)]
    explanation = explainer(sample, max_evals=2 * len(names) + 1, silent=True)
    importance = dict(zip(names, np.abs(explanation.values).mean(axis=0).tolist()))
    reports = ROOT / "reports"
    plots = reports / "visualizations"
    plots.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    shap.summary_plot(explanation.values, sample, feature_names=names, max_display=12, show=False)
    plt.savefig(plots / "shap-summary.png", bbox_inches="tight")
    plt.close()
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    path = models / "fraudshield.joblib"
    joblib.dump({"model": model, "preprocessor": prep, "background": background}, path, compress=3)
    version = "fs1-" + hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    config = {
        "model": winner, "version": version, "threshold": cut, "features": FIELDS,
        "numeric_bounds": BOUNDS, "shap_scale": "fraud score (0–1)",
        "categories": {k: sorted(train[k].unique()) for k in CAT},
        "engineered_features": ENGINEERED,
        "split": split_info, "validation": results, "test": test_metrics,
        "roc": {"FPR": fpr.tolist(), "TPR": tpr.tolist()},
        "pr": {"Recall": r.tolist(), "Precision": p.tolist()},
        "importance": importance, "training_evidence": evidence(train),
    }
    (models / "metadata.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(winner, test_metrics)


if __name__ == "__main__":
    run()
