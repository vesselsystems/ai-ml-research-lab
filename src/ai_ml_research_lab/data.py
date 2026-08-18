"""Dataset acquisition and feature preparation."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
TARGET = "Churn"
ID_COLUMN = "customerID"


def download_dataset(destination: Path, url: str = DATA_URL) -> Path:
    """Download the public source CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:  # noqa: S310 - project constant
        destination.write_bytes(response.read())
    return destination


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the source and normalize the one known numeric field with blanks."""
    frame = pd.read_csv(path)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Remove the identifier and create a binary churn target."""
    if TARGET not in frame:
        raise ValueError(f"Expected target column {TARGET!r}")

    features = frame.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    target = frame[TARGET].map({"Yes": 1, "No": 0})
    if target.isna().any():
        raise ValueError("Target contains values other than Yes/No")
    return features, target.astype("int64")
