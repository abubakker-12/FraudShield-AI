"""Plain-language field names and values shared by the workspace and evidence reports."""
from .features import CAT

LABELS = {
    "transaction_id": "Transaction ID", "hour_of_day": "Transaction Hour",
    "is_international": "Location Type", "failed_attempts": "Failed Attempts",
    "pin_changed_recently": "Recent PIN Change",
    "time_since_last_txn_hrs": "Time Since Previous Transaction",
    "transaction_amount": "Transaction Amount", "merchant_category": "Merchant Category",
    "payment_method": "Payment Method", "device_type": "Device Type",
    "international_repeat_failures": "International Activity × Repeated Failures",
    "repeat_failures_short_interval": "Repeated Failures × Short Interval",
    "log_amount_repeat_failures": "Log Amount × Repeated Failures",
}


def label(name):
    for field in CAT:
        if name.startswith(field + "_"):
            return f"{LABELS[field]} · {name[len(field) + 1:]}"
    return LABELS.get(name, name.replace("_", " ").title())


def display(name, value):
    if name == "is_international":
        return "International" if value else "Domestic"
    if name == "pin_changed_recently":
        return "Yes" if value else "No"
    if name == "hour_of_day":
        return f"{int(value):02d}:00"
    if name == "time_since_last_txn_hrs":
        return f"{value:,.2f} h"
    if name == "transaction_amount":
        return f"{value:,.2f}"
    return str(value)
