"""Build compact historical chart aggregates from the private source."""
import json

import numpy as np
import pandas as pd

from .features import RAW, ROOT


def run():
    raw = pd.read_csv(RAW, usecols=["transaction_amount", "is_fraud", "failed_attempts", "is_international"])
    bins = [0, 25, 50, 100, 200, 400, 1000, 5000, 10000, np.inf]
    raw["amount_band"] = pd.cut(raw.transaction_amount, bins, right=False)
    amounts = raw.groupby(["amount_band", "is_fraud"], observed=True).size().reset_index(name="total")
    amounts["amount_band"] = amounts["amount_band"].astype(str)
    report = {"amount_distribution": amounts.to_dict("records")}
    for key in ("failed_attempts", "is_international"):
        report[key] = raw.groupby(key).is_fraud.agg(total="size", fraud="sum", rate="mean").reset_index().to_dict("records")
    (ROOT / "reports" / "intelligence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
