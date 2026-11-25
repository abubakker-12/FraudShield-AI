"""Validate uploaded CSV data before model inference."""
import csv
from io import BytesIO, StringIO

import pandas as pd

from .features import FIELDS, features


def read_upload(data):
    if not 0 < len(data) <= 2_000_000:
        raise ValueError("Upload a nonempty CSV smaller than 2 MB.")
    try:
        text = data.decode("utf-8-sig")
        rows = csv.reader(StringIO(text), strict=True)
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
            if count > 1000:
                raise ValueError("Upload at most 1,000 transactions.")
            if len(row) != len(header):
                raise ValueError(f"CSV line {rows.line_num} has an incorrect number of fields.")
        if not count:
            raise ValueError("Provide at least one transaction.")
        frame = pd.read_csv(BytesIO(data), usecols=FIELDS, nrows=1000)
    except (UnicodeError, csv.Error, pd.errors.ParserError):
        raise ValueError("Could not read a well-formed UTF-8 CSV.") from None
    features(frame)
    return frame
