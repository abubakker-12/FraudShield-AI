"""Validate uploaded CSV data before model inference."""
import csv
from io import BytesIO, TextIOWrapper

import numpy as np
import pandas as pd

from .features import CAT, FIELDS, validate_features

UPLOAD_BYTES = 5_000_000
UPLOAD_ROWS = 25_000
INFERENCE_ROWS = 8_192


def read_upload(data, *, max_bytes=UPLOAD_BYTES, max_rows=UPLOAD_ROWS):
    if not 0 < len(data) <= max_bytes:
        raise ValueError(f"Upload a nonempty CSV no larger than {max_bytes / 1_000_000:g} MB.")
    if b"\x00" in data:
        raise ValueError("CSV contains a null byte.")
    stream = BytesIO(data)
    try:
        # Bounded text decoding retains strict CSV structure checks without a full Unicode copy.
        with TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text:
            rows = csv.reader(text, strict=True)
            header = next(rows, [])
            if len(set(header)) != len(header):
                raise ValueError("CSV column names must be unique.")
            missing = [name for name in FIELDS if name not in header]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}.")
            count = 0
            for row in rows:
                if not row:
                    continue
                count += 1
                if count > max_rows:
                    raise ValueError(f"Upload at most {max_rows:,} transactions.")
                if len(row) != len(header):
                    raise ValueError(f"CSV line {rows.line_num} has an incorrect number of fields.")
        if not count:
            raise ValueError("Provide at least one transaction.")
        frame = pd.read_csv(BytesIO(data), usecols=FIELDS)
    except (UnicodeError, csv.Error, pd.errors.ParserError):
        raise ValueError("Could not read a well-formed UTF-8 CSV.") from None
    if len(frame) != count:
        raise ValueError("CSV row count differs after parsing.")
    return validate_features(frame).astype({
        **dict.fromkeys(CAT, "category"), "hour_of_day": "uint8", "is_international": "uint8",
        "failed_attempts": "uint16", "pin_changed_recently": "uint8",
    })


def score_upload(scorer, frame, *, chunk_size=INFERENCE_ROWS, on_progress=None):
    scores = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        end = min(start + chunk_size, len(frame))
        scores[start:end] = scorer.score_validated_batch(frame.iloc[start:end])
        if on_progress is not None:
            on_progress(end)
    return scores


def prediction_frame(scores, metadata, indexes=None):
    if indexes is None:
        indexes = np.arange(len(scores))
    risk = scores[indexes].astype(float)
    threshold = metadata["threshold"]
    flags = risk >= threshold
    return pd.DataFrame({
        "Source record": indexes + 1, "risk": risk, "threshold": threshold, "suspected_fraud": flags,
        "model_version": metadata["version"],
        "Prediction": np.where(flags, "Suspected fraud", "Likely legitimate"),
    })


def export_predictions(scores, metadata):
    # Export contains only numeric fields, fixed labels and the trusted model version.
    return prediction_frame(scores, metadata).to_csv(index=False)
