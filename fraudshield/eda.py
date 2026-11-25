"""Run with python -m fraudshield.eda to reproduce the source audit."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency

from .features import RAW, ROOT


def read_source():
    df = pd.read_csv(RAW)
    stamp = pd.to_datetime(df.transaction_date + " " + df.transaction_time,
                           format="%d/%m/%Y %H:%M:%S", errors="raise")
    if not set(df.is_fraud.unique()) == {0, 1}:
        raise ValueError("Expected a binary target with both classes")
    return df, stamp


def groups(df, names):
    return (df.groupby(names, observed=True).is_fraud.agg(total="size", fraud="sum", rate="mean")
            .reset_index().to_dict("records"))


def evidence(df):
    return groups(df.assign(repeat_failures=df.failed_attempts >= 2), ["repeat_failures"])


def run():
    df, stamp = read_source()
    report = ROOT / "reports"
    plots = report / "visualizations"
    plots.mkdir(parents=True, exist_ok=True)
    amount = df.transaction_amount
    q1, q3 = amount.quantile([.25, .75])
    limit = q3 + 1.5 * (q3 - q1)
    interactions = df.assign(repeat_failures=df.failed_attempts >= 2,
                             short_interval=df.time_since_last_txn_hrs <= .5,
                             high_amount=amount > limit)
    numeric = df.select_dtypes(include="number")
    a, b = (amount[df.is_fraud.eq(i)] for i in (0, 1))
    u = mannwhitneyu(b, a, method="asymptotic")
    chi = chi2_contingency(pd.crosstab(df.is_international, df.is_fraud))
    facts = {
        "rows": len(df), "columns": len(df.columns), "fraud": int(df.is_fraud.sum()),
        "fraud_rate": float(df.is_fraud.mean()), "schema": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(), "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df.transaction_id.duplicated().sum()),
        "customers": int(df.customer_id.nunique()),
        "dates": [str(stamp.min()), str(stamp.max())],
        "hour_mismatches": int(stamp.dt.hour.ne(df.hour_of_day).sum()),
        "label_leakage": pd.crosstab(df.is_fraud, df.fraud_type.isna()).to_dict(),
        "amount": amount.describe().to_dict(),
        "amount_outliers": {"upper_fence": limit, "groups": groups(interactions, ["high_amount"])},
        "numerical_by_class": numeric.drop(columns="is_fraud").groupby(df.is_fraud).agg(
            ["mean", "median", "std", "min", "max"]).stack(level=0).reset_index().to_dict("records"),
        "categories": {key: groups(df, key) for key in
                       ["country", "city", "merchant_category", "payment_method", "device_type"]},
        "hourly": groups(df, "hour_of_day"),
        "interactions": {name: groups(interactions, cols) for name, cols in {
            "international_and_failures": ["is_international", "repeat_failures"],
            "failures_and_interval": ["repeat_failures", "short_interval"],
            "amount_and_risk": ["high_amount", "repeat_failures", "is_international"],
        }.items()},
        "statistical_comparisons": {
            "amount_mann_whitney_p": float(u.pvalue),
            "amount_probability_of_superiority": float(u.statistic / (len(a) * len(b))),
            "international_chi_square": float(chi.statistic),
            "international_chi_square_p": float(chi.pvalue),
        },
    }
    corr = numeric.corr()
    corr.to_csv(report / "correlation.csv")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlations")
    fig.savefig(plots / "correlation.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sample = df.sample(60000, random_state=42)
    sns.histplot(sample.assign(log_amount=np.log1p(sample.transaction_amount)),
                 x="log_amount", hue="is_fraud", stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title("Amount distribution by recorded class")
    rates = pd.DataFrame(facts["interactions"]["international_and_failures"])
    sns.barplot(rates, x="is_international", y="rate", hue="repeat_failures", ax=axes[1])
    axes[1].set_title("Recorded fraud rate by location and repeated failures")
    fig.savefig(plots / "distributions.png", bbox_inches="tight")
    plt.close(fig)
    (report / "eda.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"Audited {len(df):,} rows. See reports/eda.json.")


if __name__ == "__main__":
    run()
