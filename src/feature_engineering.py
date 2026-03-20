"""
Feature engineering module for SaaS Account Churn Prediction.

Creates SaaS-specific derived features, encodes categoricals, and scales
numeric features for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REFERENCE_DATE = pd.Timestamp("2025-12-01")

NUMERIC_FEATURES = [
    "mrr",
    "arr",
    "seats_purchased",
    "seats_active",
    "seat_utilization_pct",
    "feature_adoption_score",
    "days_to_activation",
    "dau_mau_ratio",
    "nps_score",
    "support_tickets_open",
    "support_tickets_severity_avg",
    "last_login_days_ago",
    "api_calls_last_30d",
    "expansion_mrr",
    "contraction_mrr",
    "net_revenue_retention_pct",
    "health_score",
    "days_since_nps",
    "contract_renewal_proximity",
]


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def compute_health_score(df: pd.DataFrame) -> pd.Series:
    """
    Weighted composite health score (0-100):
      0.30 × feature_adoption_score
    + 0.25 × (dau_mau_ratio × 100)
    + 0.25 × seat_utilization_pct
    + 0.20 × (100 − support_tickets_severity_avg × 20)
    """
    support_component = np.clip(100 - df["support_tickets_severity_avg"] * 20, 0, 100)
    score = (
        0.30 * df["feature_adoption_score"]
        + 0.25 * (df["dau_mau_ratio"] * 100)
        + 0.25 * df["seat_utilization_pct"]
        + 0.20 * support_component
    )
    return np.clip(score, 0, 100).round(1)


def compute_days_since_nps(df: pd.DataFrame,
                           reference_date: pd.Timestamp = REFERENCE_DATE) -> pd.Series:
    """Days since the last NPS response."""
    nps_date = pd.to_datetime(df["nps_response_date"])
    return (reference_date - nps_date).dt.days.astype(float)


def compute_nps_risk_flag(df: pd.DataFrame) -> pd.Series:
    """1 if NPS score < 0 (detractor)."""
    return (df["nps_score"] < 0).astype(int)


def compute_expansion_flag(df: pd.DataFrame) -> pd.Series:
    """1 if expansion_mrr > 0 (account expanded recently)."""
    return (df["expansion_mrr"] > 0).astype(int)


def compute_engagement_trend(df: pd.DataFrame) -> pd.Series:
    """
    Ratio proxy for API engagement trend.

    Since we only have api_calls_last_30d, we create a synthetic prior-period
    value using a jittered baseline to simulate an engagement trend signal.
    In production, this would come from two actual time windows.
    """
    # Proxy: compare current API calls vs a baseline derived from MRR tier
    plan_baseline = df["api_calls_last_30d"].median()
    baseline = np.where(plan_baseline > 0, plan_baseline, 1)
    trend = (df["api_calls_last_30d"] / baseline).round(3)
    return trend.astype(float)


def compute_contract_renewal_proximity(
    df: pd.DataFrame,
    reference_date: pd.Timestamp = REFERENCE_DATE,
) -> pd.Series:
    """
    Days until contract renewal.

    Calculated as: (signup_date + contract_length in months) rolled forward
    to the next renewal cycle relative to reference_date.
    """
    signup = pd.to_datetime(df["signup_date"])
    contract_months = df["contract_length_months"]

    results = []
    for s, cm in zip(signup, contract_months):
        # Roll forward through renewal cycles
        renewal = s + pd.DateOffset(months=int(cm))
        while renewal <= reference_date:
            renewal += pd.DateOffset(months=int(cm))
        days_until = (renewal - reference_date).days
        results.append(days_until)

    return pd.Series(results, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Aggregate pipeline
# ---------------------------------------------------------------------------


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived SaaS features to the DataFrame."""
    df = df.copy()
    df["health_score"] = compute_health_score(df)
    df["days_since_nps"] = compute_days_since_nps(df)
    df["nps_risk_flag"] = compute_nps_risk_flag(df)
    df["expansion_flag"] = compute_expansion_flag(df)
    df["engagement_trend"] = compute_engagement_trend(df)
    df["contract_renewal_proximity"] = compute_contract_renewal_proximity(df)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical SaaS features."""
    df = df.copy()
    cat_cols = ["plan_type", "acquisition_channel"]

    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

    # Encode contract_length_months as dummies (it's categorical in nature)
    if "contract_length_months" in df.columns:
        dummies = pd.get_dummies(
            df["contract_length_months"].astype(str),
            prefix="contract",
            drop_first=True,
            dtype=int,
        )
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=["contract_length_months"])

    return df


def scale_features(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler on numeric features."""
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def prepare_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Full feature pipeline: engineer → encode → drop non-model cols → scale."""
    df = add_engineered_features(df)
    df = encode_categoricals(df)

    # Drop columns not needed for modelling
    drop_cols = [c for c in ["signup_date", "nps_response_date", "company_name"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df, scaler = scale_features(df, scaler=scaler, fit=fit)
    return df, scaler
