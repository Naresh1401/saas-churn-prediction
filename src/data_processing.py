"""
Data processing module for SaaS Account Churn Prediction.

Handles loading, cleaning, and splitting the raw accounts dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw SaaS accounts CSV."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "raw" / "saas_accounts.csv"
    df = pd.read_csv(
        path,
        parse_dates=["signup_date", "nps_response_date"],
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values and invalid entries."""
    df = df.copy()

    # Fill missing numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Parse dates
    for col in ["signup_date", "nps_response_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Drop duplicate account_ids
    df = df.drop_duplicates(subset="account_id", keep="last")

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "logo_churn_flag",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split into train/test sets, stratified on the target."""
    drop_cols = [target_col, "account_id", "company_name"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def save_processed(df: pd.DataFrame, filename: str = "accounts_clean.csv"):
    """Save processed data to data/processed/."""
    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / filename, index=False)
