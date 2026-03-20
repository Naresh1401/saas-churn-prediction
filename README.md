# SaaS Account Churn Prediction

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-teal)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

> End-to-end ML pipeline that predicts **logo churn** for B2B SaaS accounts, generates a composite **Health Score**, and recommends **CSM playbook actions** — served through a production-ready REST API.

---

## Business Context

| Metric | Impact |
|---|---|
| Logo churn | Lost accounts = lost ARR + negative signal to investors |
| Net Revenue Retention (NRR) | Best-in-class SaaS cos. target NRR > 120% |
| Customer Health Score | Proactive leading indicator vs. reactive lagging metrics |
| CSM Playbook | Operationalise model output into repeatable retention workflows |

This project targets a **mid-level Data Science / ML Engineer** portfolio and interview
preparation for B2B SaaS companies.

---

## Architecture

```
saas_accounts.csv
        │
        ▼
┌─────────────────┐     ┌───────────────────┐     ┌───────────────┐
│  generate_data   │────▶│  data_processing   │────▶│  feature_eng   │
│  (5 000 accts)   │     │  clean + split     │     │  health_score  │
└─────────────────┘     └───────────────────┘     │  nps_risk_flag │
                                                    │  expansion_flag│
                                                    │  renewal_prox  │
                                                    └───────┬───────┘
                                                            │
                                                            ▼
                        ┌───────────────────────────────────────────┐
                        │  train.py                                  │
                        │  SMOTE ➜ GridSearchCV (LR / RF / XGBoost) │
                        │  SHAP explanations + health score plot     │
                        └───────────────────┬───────────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────────┐
                        │  FastAPI /predict                          │
                        │  churn_probability │ risk_tier │ actions   │
                        └───────────────────────────────────────────┘
```

---

## SaaS Metrics Glossary

| Term | Definition |
|---|---|
| **MRR** | Monthly Recurring Revenue |
| **ARR** | Annual Recurring Revenue (MRR × 12) |
| **NRR** | Net Revenue Retention — expansion minus contraction & churn |
| **Logo Churn** | Binary: did the account cancel? (vs. revenue churn) |
| **DAU/MAU** | Daily Active Users ÷ Monthly Active Users — product stickiness |
| **Seat Utilization** | Active seats ÷ purchased seats |
| **Feature Adoption Score** | % of core product features actively used |
| **CSM** | Customer Success Manager |
| **QBR** | Quarterly Business Review |
| **NPS** | Net Promoter Score (−100 to +100) |

---

## Engineered Features

| Feature | Formula / Logic |
|---|---|
| `health_score` | 0.30 × feature_adoption + 0.25 × (DAU/MAU × 100) + 0.25 × seat_util + 0.20 × (100 − severity × 20) |
| `nps_risk_flag` | 1 if NPS < 0 (detractor) |
| `expansion_flag` | 1 if expansion_mrr > 0 |
| `engagement_trend` | api_calls_last_30d ÷ median(api_calls_last_30d) |
| `contract_renewal_proximity` | Days until next renewal cycle |
| `days_since_nps` | Days from last NPS response to reference date |

---

## Risk Tiers & CSM Playbook

| Risk Tier | Threshold | Recommended Actions |
|---|---|---|
| **High** | churn_prob ≥ 0.6 | Emergency QBR, exec sponsor escalation, custom training, discount offer |
| **Medium** | 0.3 ≤ churn_prob < 0.6 | Schedule QBR, assign CSM if none, feature adoption workshop |
| **Low** | churn_prob < 0.3 | Standard check-in cadence, explore expansion / upsell |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Naresh1401/saas-churn-prediction.git
cd saas-churn-prediction

# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pipeline (or `make all`)
make data      # generate 5 000 synthetic accounts
make train     # SMOTE + GridSearchCV + SHAP
make test      # 15+ pytest assertions

# API
make api       # uvicorn on port 8000
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "ACCT-042",
    "plan_type": "growth",
    "mrr": 5000, "arr": 60000,
    "seats_purchased": 50, "seats_active": 30,
    "seat_utilization_pct": 60,
    "feature_adoption_score": 45,
    "days_to_activation": 10,
    "dau_mau_ratio": 0.35,
    "nps_score": 15,
    "support_tickets_open": 5,
    "support_tickets_severity_avg": 2.5,
    "last_login_days_ago": 12,
    "api_calls_last_30d": 500,
    "expansion_mrr": 200, "contraction_mrr": 0,
    "net_revenue_retention_pct": 105,
    "csm_assigned": 1, "qbr_completed": 1,
    "acquisition_channel": "inbound",
    "contract_length_months": 12,
    "signup_date": "2024-03-15",
    "nps_response_date": "2025-05-01"
  }'
```

---

## Project Structure

```
saas-churn-prediction/
├── api/
│   └── main.py                 # FastAPI app — /health, /predict
├── data/
│   ├── raw/                    # generated CSV
│   └── processed/              # train/test splits
├── models/                     # best_model.pkl
├── notebooks/
│   └── 01_EDA.ipynb            # churn EDA + SaaS-specific insights
├── outputs/                    # metrics.json, SHAP plot, health score plot
├── src/
│   ├── generate_data.py        # synthetic B2B SaaS account generator
│   ├── data_processing.py      # load, clean, split
│   ├── feature_engineering.py  # health score, NPS flags, engagement trend
│   ├── train.py                # SMOTE + GridSearchCV + SHAP + plots
│   └── predict.py              # single-account prediction with CSM actions
├── tests/
│   └── test_features.py        # unit + integration tests
├── .github/workflows/ci.yml    # GitHub Actions CI
├── Makefile                    # one-command targets
├── requirements.txt            # pinned deps
└── README.md
```

---

## Model Performance

*Model metrics are written to `outputs/metrics.json` after training. Typical results:*

| Model | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.77 | ~0.54 | ~0.42 | ~0.75 |
| Random Forest | ~0.78 | ~0.52 | ~0.44 | ~0.64 |
| **XGBoost** | **~0.79** | **~0.55** | **~0.44** | **~0.74** |

> Best model selected automatically by ROC-AUC on the holdout set.

---

## License

MIT
