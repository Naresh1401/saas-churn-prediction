"""
FastAPI application for SaaS Account Churn Prediction.

Exposes a POST /predict endpoint that returns churn probability,
risk tier, health score, SHAP-based risk factors, and recommended CSM
actions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict_single

app = FastAPI(
    title="SaaS Account Churn Prediction API",
    version="1.0.0",
    description="Predict B2B SaaS account churn with health scoring and explainable AI.",
)

_model, _scaler, _feature_names = None, None, None


@app.on_event("startup")
def startup_load_model():
    global _model, _scaler, _feature_names
    _model, _scaler, _feature_names = load_model()


# --------------- Schemas ---------------


class PredictRequest(BaseModel):
    account_id: str
    features: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "account_id": "ACC-00042",
                    "features": {
                        "signup_date": "2023-03-15",
                        "plan_type": "starter",
                        "mrr": 350.0,
                        "arr": 4200.0,
                        "seats_purchased": 8,
                        "seats_active": 3,
                        "seat_utilization_pct": 37.5,
                        "feature_adoption_score": 22.0,
                        "days_to_activation": 35,
                        "dau_mau_ratio": 0.12,
                        "nps_score": -15,
                        "nps_response_date": "2025-09-01",
                        "support_tickets_open": 5,
                        "support_tickets_severity_avg": 3.8,
                        "last_login_days_ago": 42,
                        "api_calls_last_30d": 120,
                        "expansion_mrr": 0,
                        "contraction_mrr": 50.0,
                        "net_revenue_retention_pct": 85.7,
                        "csm_assigned": 0,
                        "qbr_completed": 0,
                        "acquisition_channel": "self-serve",
                        "contract_length_months": 1,
                    },
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    account_id: str
    churn_probability: float
    risk_tier: str
    health_score: float
    top_risk_factors: list[str]
    recommended_action: str


# --------------- Endpoints ---------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict churn probability for a single SaaS account."""
    try:
        result = predict_single(
            features=request.features,
            model=_model,
            scaler=_scaler,
            feature_names=_feature_names,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        account_id=request.account_id,
        churn_probability=result["churn_probability"],
        risk_tier=result["risk_tier"],
        health_score=result["health_score"],
        top_risk_factors=result["top_risk_factors"],
        recommended_action=result["recommended_action"],
    )
