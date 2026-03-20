"""
Prediction module for SaaS Account Churn Prediction.

Loads the saved model artefact and provides a predict function used by
the FastAPI endpoint and batch scoring.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.feature_engineering import prepare_features, compute_health_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_model(path: Path | str | None = None):
    """Load the persisted model artefact (model, scaler, feature_names)."""
    if path is None:
        path = MODEL_PATH
    artifact = joblib.load(path)
    return artifact["model"], artifact["scaler"], artifact["feature_names"]


# ---------------------------------------------------------------------------
# Recommended action logic
# ---------------------------------------------------------------------------

_ACTIONS = {
    "High": (
        "Schedule QBR within 7 days, assign CSM, trigger onboarding "
        "re-engagement campaign, escalate to VP Customer Success"
    ),
    "Medium": (
        "Schedule CSM check-in within 14 days, review feature adoption "
        "gaps, offer training session"
    ),
    "Low": (
        "Continue monitoring, include in next quarterly outreach, "
        "consider expansion opportunity"
    ),
}


def predict_single(features: dict, model=None, scaler=None, feature_names=None):
    """
    Predict churn probability for a single SaaS account.

    Parameters
    ----------
    features : dict
        Raw feature values (before engineering / scaling).

    Returns
    -------
    dict with churn_probability, risk_tier, health_score,
         top_risk_factors, recommended_action
    """
    if model is None:
        model, scaler, feature_names = load_model()

    # Build a one-row DataFrame
    df = pd.DataFrame([features])

    # Parse dates
    for col in ["signup_date", "nps_response_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Compute health score BEFORE scaling (need raw values)
    raw_health_score = float(compute_health_score(df).iloc[0])

    # Feature pipeline
    df_feat, _ = prepare_features(df, scaler=scaler, fit=False)

    # Align columns
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    df_feat = df_feat[feature_names]

    # Predict
    prob = float(model.predict_proba(df_feat)[:, 1][0])

    # Risk tier
    if prob >= 0.6:
        risk_tier = "High"
    elif prob >= 0.3:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # SHAP-based risk factors
    top_risk_factors = _explain(model, df_feat, feature_names, features)

    return {
        "churn_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "health_score": round(raw_health_score, 1),
        "top_risk_factors": top_risk_factors,
        "recommended_action": _ACTIONS[risk_tier],
    }


def _explain(model, df_feat: pd.DataFrame, feature_names: list[str],
             raw_features: dict) -> list[str]:
    """Return top-3 plain-English risk factors from SHAP values."""
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, df_feat)
        sv = explainer.shap_values(df_feat)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.array(sv).flatten()
    except Exception:
        return ["Unable to compute explanations for this model type."]

    # Sort by absolute impact (positive SHAP = pushes toward churn)
    impacts = sorted(
        zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True
    )

    # Map feature names → business-friendly descriptions with actual values
    def _describe(feat: str, shap_val: float) -> str:
        raw = raw_features.get(feat)
        templates = {
            "feature_adoption_score": (
                f"Low feature adoption ({raw}%)" if raw and raw < 50
                else f"Feature adoption at {raw}%"
            ),
            "seat_utilization_pct": (
                f"Low seat utilization ({raw}%)" if raw and raw < 40
                else f"Seat utilization at {raw}%"
            ),
            "dau_mau_ratio": (
                f"Low product stickiness (DAU/MAU: {raw})" if raw and raw < 0.25
                else f"DAU/MAU ratio: {raw}"
            ),
            "support_tickets_open": (
                f"High open support tickets ({raw})" if raw and raw > 3
                else f"{raw} open tickets"
            ),
            "support_tickets_severity_avg": f"High ticket severity (avg {raw}/5)",
            "last_login_days_ago": (
                f"No recent login ({raw} days)" if raw and raw > 30
                else f"Last login {raw} days ago"
            ),
            "nps_score": (
                f"NPS detractor (score: {raw})" if raw and raw < 0
                else f"NPS score: {raw}"
            ),
            "nps_risk_flag": "NPS detractor" if shap_val > 0 else "NPS promoter/passive",
            "csm_assigned": "No CSM assigned" if raw == 0 else "CSM assigned",
            "qbr_completed": "No QBR completed" if raw == 0 else "QBR completed",
            "health_score": (
                f"Low health score ({raw})" if raw and raw < 40
                else f"Health score: {raw}"
            ),
            "expansion_flag": (
                "No recent expansion" if shap_val > 0
                else "Recent expansion detected"
            ),
            "expansion_mrr": (
                "No expansion revenue" if raw == 0
                else f"${raw} expansion MRR"
            ),
            "contraction_mrr": (
                f"Revenue contraction (${raw})" if raw and raw > 0
                else "No contraction"
            ),
            "net_revenue_retention_pct": (
                f"Low NRR ({raw}%)" if raw and raw < 100
                else f"NRR at {raw}%"
            ),
            "contract_renewal_proximity": f"Contract renewal in {raw} days",
            "mrr": f"MRR: ${raw}",
            "days_to_activation": (
                f"Slow activation ({raw} days)" if raw and raw > 30
                else f"Activated in {raw} days"
            ),
            "engagement_trend": (
                f"Declining engagement (trend: {raw})" if raw and raw < 0.8
                else f"Engagement trend: {raw}"
            ),
        }
        desc = templates.get(feat, f"{feat}: {raw}")
        direction = "increases" if shap_val > 0 else "decreases"
        return f"{desc} — {direction} churn risk"

    reasons = []
    for feat, val in impacts[:3]:
        if abs(val) > 0.001:
            reasons.append(_describe(feat, val))

    return reasons if reasons else ["No dominant risk factors identified."]
