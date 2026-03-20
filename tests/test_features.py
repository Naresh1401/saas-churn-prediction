"""Tests for feature engineering, data processing, and API."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    compute_health_score,
    compute_days_since_nps,
    compute_nps_risk_flag,
    compute_expansion_flag,
    compute_engagement_trend,
    compute_contract_renewal_proximity,
)


# ---------- Feature engineering unit tests ----------

class TestHealthScore:
    def test_returns_between_0_and_100(self):
        df = pd.DataFrame({
            "feature_adoption_score": [80, 10, 50, 0, 100],
            "dau_mau_ratio": [0.6, 0.1, 0.3, 0.0, 1.0],
            "seat_utilization_pct": [90, 20, 50, 0, 100],
            "support_tickets_severity_avg": [1.0, 4.0, 2.5, 5.0, 0.0],
        })
        result = compute_health_score(df)
        assert (result >= 0).all() and (result <= 100).all()

    def test_high_engagement_gives_high_score(self):
        df = pd.DataFrame({
            "feature_adoption_score": [95],
            "dau_mau_ratio": [0.9],
            "seat_utilization_pct": [95],
            "support_tickets_severity_avg": [0.5],
        })
        assert compute_health_score(df).iloc[0] > 70

    def test_low_engagement_gives_low_score(self):
        df = pd.DataFrame({
            "feature_adoption_score": [5],
            "dau_mau_ratio": [0.05],
            "seat_utilization_pct": [5],
            "support_tickets_severity_avg": [4.5],
        })
        assert compute_health_score(df).iloc[0] < 30


class TestNPSRiskFlag:
    def test_detractors_flagged(self):
        df = pd.DataFrame({"nps_score": [-30, 0, 50, -5, 80]})
        result = compute_nps_risk_flag(df)
        assert result.tolist() == [1, 0, 0, 1, 0]


class TestExpansionFlag:
    def test_positive_expansion(self):
        df = pd.DataFrame({"expansion_mrr": [100, 0, -50, 250]})
        result = compute_expansion_flag(df)
        assert result.tolist() == [1, 0, 0, 1]


class TestEngagementTrend:
    def test_above_median_is_positive(self):
        df = pd.DataFrame({"api_calls_last_30d": [100, 200, 300, 400, 500]})
        result = compute_engagement_trend(df)
        assert result.iloc[4] > 1.0
        assert result.iloc[0] < 1.0


class TestContractRenewalProximity:
    def test_month_to_month_max_30(self):
        df = pd.DataFrame({
            "signup_date": pd.to_datetime(["2025-01-15"]),
            "contract_length_months": [1],
        })
        result = compute_contract_renewal_proximity(df)
        assert 0 <= result.iloc[0] <= 31


class TestDaysSinceNPS:
    def test_returns_positive_for_past_date(self):
        df = pd.DataFrame({
            "nps_response_date": pd.to_datetime(["2025-06-01"]),
        })
        result = compute_days_since_nps(df)
        assert result.iloc[0] >= 0

    def test_missing_returns_nan(self):
        df = pd.DataFrame({"nps_response_date": [pd.NaT]})
        result = compute_days_since_nps(df)
        assert np.isnan(result.iloc[0])


# ---------- API Integration Tests ----------

@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


SAMPLE_PAYLOAD = {
    "account_id": "TEST-001",
    "features": {
        "plan_type": "growth",
        "mrr": 5000.0,
        "arr": 60000.0,
        "seats_purchased": 50,
        "seats_active": 30,
        "seat_utilization_pct": 60.0,
        "feature_adoption_score": 45.0,
        "days_to_activation": 10,
        "dau_mau_ratio": 0.35,
        "nps_score": 15,
        "support_tickets_open": 5,
        "support_tickets_severity_avg": 2.5,
        "last_login_days_ago": 12,
        "api_calls_last_30d": 500,
        "expansion_mrr": 200.0,
        "contraction_mrr": 0.0,
        "net_revenue_retention_pct": 105.0,
        "csm_assigned": 1,
        "qbr_completed": 1,
        "acquisition_channel": "inbound",
        "contract_length_months": 12,
        "signup_date": "2024-03-15",
        "nps_response_date": "2025-05-01",
    },
}


class TestHealthEndpoint:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_returns_200(self, api_client):
        resp = api_client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200

    def test_response_keys(self, api_client):
        resp = api_client.post("/predict", json=SAMPLE_PAYLOAD)
        body = resp.json()
        required = {
            "account_id", "churn_probability", "risk_tier",
            "health_score", "top_risk_factors", "recommended_action",
        }
        assert required.issubset(body.keys())

    def test_churn_probability_range(self, api_client):
        resp = api_client.post("/predict", json=SAMPLE_PAYLOAD)
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_tier_valid(self, api_client):
        resp = api_client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.json()["risk_tier"] in ("High", "Medium", "Low")

    def test_health_score_range(self, api_client):
        resp = api_client.post("/predict", json=SAMPLE_PAYLOAD)
        hs = resp.json()["health_score"]
        assert 0.0 <= hs <= 100.0

    def test_missing_field_returns_422(self, api_client):
        payload = {"account_id": "TEST-001"}  # missing 'features'
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 422
