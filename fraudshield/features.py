from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Dataset" / "bank_fraud.csv"
CAT = ["merchant_category", "payment_method", "device_type"]
BOUNDS = {
    "hour_of_day": (0, 23), "is_international": (0, 1),
    "failed_attempts": (0, 1000), "pin_changed_recently": (0, 1),
    "time_since_last_txn_hrs": (0, 1e6), "transaction_amount": (0.01, 1e12),
}
NUM = list(BOUNDS)
FIELDS = NUM + CAT
ENGINEERED = ["overnight", "repeat_failures", "short_interval",
              "international_repeat_failures", "repeat_failures_short_interval",
              "log_amount", "log_amount_repeat_failures"]


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(validate_features(frame))


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FIELDS) - set(frame))
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    x = frame[FIELDS].copy()
    for name, (lower, upper) in BOUNDS.items():
        values = pd.to_numeric(x[name], errors="coerce").astype(float)
        valid = np.isfinite(values) & values.between(lower, upper)
        if name in NUM[:4]:
            valid &= values.mod(1).eq(0)
        if not valid.all():
            raise ValueError(f"Invalid values for {name} in {(~valid).sum()} row(s)")
        x[name] = values
    for name in CAT:
        values = x[name].astype("string").str.strip()
        if not values.str.len().between(1, 80).fillna(False).all():
            raise ValueError(f"Invalid category for {name}")
        x[name] = values.astype(object)
    return x


def engineer_features(validated: pd.DataFrame) -> pd.DataFrame:
    # Assign returns a separate frame, preserving the selected source rows for the investigation workflow.
    x = validated.assign(overnight=(validated.hour_of_day.lt(7) | validated.hour_of_day.ge(22)).astype(int))
    x["repeat_failures"] = x.failed_attempts.ge(2).astype(int)
    x["short_interval"] = x.time_since_last_txn_hrs.le(0.5).astype(int)
    x["international_repeat_failures"] = x.is_international * x.repeat_failures
    x["repeat_failures_short_interval"] = x.repeat_failures * x.short_interval
    x["log_amount"] = np.log1p(x.transaction_amount)
    x["log_amount_repeat_failures"] = x.log_amount * x.repeat_failures
    return x
