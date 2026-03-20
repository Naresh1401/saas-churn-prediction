"""
Synthetic B2B SaaS Dataset Generator.

Generates 5,000 account-level records with realistic SaaS metrics
and ~18% logo churn rate with embedded domain-specific correlations.
"""

import numpy as np
import pandas as pd
from pathlib import Path


COMPANY_PREFIXES = [
    "Acme", "Apex", "Nova", "Vero", "Zeta", "Pulse", "Orbit", "Nexus",
    "Crest", "Forge", "Peak", "Flux", "Core", "Edge", "Link", "Wave",
    "Grid", "Sync", "Loop", "Bolt", "Arch", "Hive", "Helm", "Sage",
]
COMPANY_SUFFIXES = [
    "Tech", "Labs", "AI", "Cloud", "Systems", "Solutions", "Digital",
    "Software", "Data", "Analytics", "Platform", "Corp", "Inc", "HQ",
    "IO", "Works", "Group", "Dynamics", "Logic", "Stack",
]


def generate_synthetic_data(n_accounts: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic B2B SaaS churn dataset."""
    rng = np.random.default_rng(seed)

    # --- Identifiers ---
    account_ids = [f"ACC-{i:05d}" for i in range(1, n_accounts + 1)]
    company_names = [
        f"{rng.choice(COMPANY_PREFIXES)} {rng.choice(COMPANY_SUFFIXES)}"
        for _ in range(n_accounts)
    ]

    # --- Signup dates: 2021-01-01 to 2025-06-30 ---
    signup_start = pd.Timestamp("2021-01-01")
    signup_range = (pd.Timestamp("2025-06-30") - signup_start).days
    signup_offsets = rng.integers(0, signup_range, size=n_accounts)
    signup_dates = [signup_start + pd.Timedelta(days=int(d)) for d in signup_offsets]

    # --- Plan type ---
    plan_types = rng.choice(
        ["starter", "growth", "enterprise"],
        size=n_accounts,
        p=[0.40, 0.38, 0.22],
    )

    # --- Acquisition channel ---
    acquisition_channels = rng.choice(
        ["inbound", "outbound", "partner", "self-serve"],
        size=n_accounts,
        p=[0.30, 0.20, 0.15, 0.35],
    )

    # --- Contract length ---
    contract_map = {"starter": [1, 12], "growth": [12, 24], "enterprise": [12, 24]}
    contract_probs = {"starter": [0.6, 0.4], "growth": [0.5, 0.5], "enterprise": [0.2, 0.8]}
    contract_length = np.array([
        rng.choice(contract_map[p], p=contract_probs[p]) for p in plan_types
    ])

    # --- CSM assignment (more likely for growth/enterprise) ---
    csm_prob_map = {"starter": 0.10, "growth": 0.55, "enterprise": 0.90}
    csm_assigned = np.array([
        rng.random() < csm_prob_map[p] for p in plan_types
    ]).astype(int)

    # --- QBR completed (only if CSM assigned, and higher for enterprise) ---
    qbr_completed = np.zeros(n_accounts, dtype=int)
    for i in range(n_accounts):
        if csm_assigned[i]:
            prob = 0.8 if plan_types[i] == "enterprise" else 0.5
            qbr_completed[i] = int(rng.random() < prob)

    # --- Seats ---
    seat_map = {"starter": (3, 8), "growth": (10, 50), "enterprise": (30, 300)}
    seats_purchased = np.array([
        rng.integers(seat_map[p][0], seat_map[p][1] + 1) for p in plan_types
    ])
    # Active seats (some fraction of purchased)
    base_util = rng.beta(4, 3, size=n_accounts)  # centered around 0.55
    seats_active = np.clip(
        (seats_purchased * base_util).astype(int), 1, seats_purchased
    )
    seat_utilization_pct = (seats_active / seats_purchased * 100).round(1)

    # --- MRR / ARR ---
    plan_mrr_map = {"starter": (200, 80), "growth": (1500, 500), "enterprise": (8000, 3000)}
    mrr = np.array([
        max(50, rng.normal(plan_mrr_map[p][0], plan_mrr_map[p][1]))
        for p in plan_types
    ]).round(2)
    arr = (mrr * 12).round(2)

    # --- Feature adoption score (0-100) ---
    plan_adoption_base = {"starter": (40, 18), "growth": (58, 15), "enterprise": (72, 12)}
    feature_adoption_score = np.array([
        np.clip(rng.normal(plan_adoption_base[p][0], plan_adoption_base[p][1]), 2, 100)
        for p in plan_types
    ]).round(1)
    # Boost adoption if CSM assigned
    feature_adoption_score = np.clip(
        feature_adoption_score + csm_assigned * rng.uniform(3, 10, n_accounts), 2, 100
    ).round(1)

    # --- Days to activation ---
    days_to_activation = np.clip(
        rng.lognormal(mean=2.5, sigma=0.8, size=n_accounts).astype(int), 1, 180
    )

    # --- DAU/MAU ratio (stickiness) ---
    dau_mau_ratio = np.clip(
        rng.beta(3, 4, size=n_accounts) * 0.8 + 0.05, 0.02, 0.80
    ).round(3)
    # Higher for enterprise with good adoption
    enterprise_mask = np.array([p == "enterprise" for p in plan_types])
    dau_mau_ratio[enterprise_mask] = np.clip(
        dau_mau_ratio[enterprise_mask] + 0.08, 0.02, 0.80
    )

    # --- NPS ---
    nps_score = np.clip(rng.normal(30, 35, size=n_accounts).astype(int), -100, 100)
    reference_date = pd.Timestamp("2025-12-01")
    nps_offsets = rng.integers(10, 200, size=n_accounts)
    nps_response_dates = [reference_date - pd.Timedelta(days=int(d)) for d in nps_offsets]

    # --- Support tickets ---
    ticket_lambda = {"starter": 2.0, "growth": 1.5, "enterprise": 1.0}
    support_tickets_open = np.array([
        rng.poisson(ticket_lambda[p]) for p in plan_types
    ])
    support_tickets_severity_avg = np.clip(
        rng.normal(2.5, 0.8, size=n_accounts), 1.0, 5.0
    ).round(1)
    # Accounts with many tickets tend to have higher severity
    high_ticket_mask = support_tickets_open > 3
    support_tickets_severity_avg[high_ticket_mask] = np.clip(
        support_tickets_severity_avg[high_ticket_mask] + 0.5, 1.0, 5.0
    )

    # --- Login recency ---
    last_login_days_ago = np.clip(
        rng.exponential(15, size=n_accounts).astype(int), 0, 120
    )

    # --- API calls ---
    api_base = {"starter": 500, "growth": 5000, "enterprise": 30000}
    api_calls_last_30d = np.array([
        max(0, int(rng.normal(api_base[p], api_base[p] * 0.4)))
        for p in plan_types
    ])

    # --- Expansion / contraction MRR ---
    expansion_mrr = np.zeros(n_accounts)
    expansion_mask = rng.random(n_accounts) < 0.25
    expansion_mrr[expansion_mask] = (
        mrr[expansion_mask] * rng.uniform(0.05, 0.3, expansion_mask.sum())
    ).round(2)

    contraction_mrr = np.zeros(n_accounts)
    contraction_mask = rng.random(n_accounts) < 0.15
    contraction_mrr[contraction_mask] = (
        mrr[contraction_mask] * rng.uniform(0.05, 0.2, contraction_mask.sum())
    ).round(2)

    # --- NRR ---
    net_revenue_retention_pct = (
        ((mrr + expansion_mrr - contraction_mrr) / mrr) * 100
    ).round(1)
    net_revenue_retention_pct = np.clip(net_revenue_retention_pct, 50, 180)

    # ===================================================================
    # Generate churn (logo_churn_flag) with realistic SaaS correlations
    # ===================================================================
    plan_bias = np.array([
        0.5 if p == "starter" else 0.0 if p == "growth" else -0.7
        for p in plan_types
    ])
    channel_bias = np.array([
        0.3 if c == "self-serve" else 0.0 if c == "inbound" else -0.1 if c == "outbound" else -0.2
        for c in acquisition_channels
    ])
    contract_bias = np.where(contract_length == 1, 0.6, np.where(contract_length == 12, 0.0, -0.4))

    logit = (
        -2.2
        - 0.04 * (feature_adoption_score - 50)
        + 0.15 * (support_tickets_open - 2)
        + 0.08 * (support_tickets_severity_avg - 2.5)
        - 2.5 * (dau_mau_ratio - 0.35)
        - 0.02 * (seat_utilization_pct - 55)
        + 0.02 * (last_login_days_ago - 15)
        - 0.6 * csm_assigned
        - 0.01 * (nps_score - 30)
        - 0.002 * (net_revenue_retention_pct - 100)
        + plan_bias
        + channel_bias
        + contract_bias
        - 0.3 * (expansion_mrr > 0).astype(float)
    )

    churn_prob = 1 / (1 + np.exp(-logit))
    logo_churn_flag = (rng.random(n_accounts) < churn_prob).astype(int)

    # Calibrate towards ~18%
    actual_rate = logo_churn_flag.mean()
    if abs(actual_rate - 0.18) > 0.02:
        from scipy.special import logit as sp_logit, expit
        shift = sp_logit(0.18) - sp_logit(max(min(actual_rate, 0.99), 0.01))
        adj_prob = expit(
            np.log(np.clip(churn_prob, 1e-9, 1 - 1e-9)
                   / (1 - np.clip(churn_prob, 1e-9, 1 - 1e-9))) + shift
        )
        logo_churn_flag = (rng.random(n_accounts) < adj_prob).astype(int)

    # --- Sprinkle ~2% missing values for realism ---
    # Convert integer arrays to float so they can hold NaN
    nps_score = nps_score.astype(float)
    feature_adoption_score = feature_adoption_score.astype(float)
    for arr_data in [nps_score, dau_mau_ratio, feature_adoption_score,
                     support_tickets_severity_avg]:
        mask = rng.random(n_accounts) < 0.02
        arr_data[mask] = np.nan

    df = pd.DataFrame({
        "account_id": account_ids,
        "company_name": company_names,
        "signup_date": signup_dates,
        "plan_type": plan_types,
        "mrr": mrr,
        "arr": arr,
        "seats_purchased": seats_purchased,
        "seats_active": seats_active,
        "seat_utilization_pct": seat_utilization_pct,
        "feature_adoption_score": feature_adoption_score,
        "days_to_activation": days_to_activation,
        "dau_mau_ratio": dau_mau_ratio,
        "nps_score": nps_score,
        "nps_response_date": nps_response_dates,
        "support_tickets_open": support_tickets_open,
        "support_tickets_severity_avg": support_tickets_severity_avg,
        "last_login_days_ago": last_login_days_ago,
        "api_calls_last_30d": api_calls_last_30d,
        "expansion_mrr": expansion_mrr,
        "contraction_mrr": contraction_mrr,
        "net_revenue_retention_pct": net_revenue_retention_pct,
        "csm_assigned": csm_assigned,
        "qbr_completed": qbr_completed,
        "acquisition_channel": acquisition_channels,
        "contract_length_months": contract_length,
        "logo_churn_flag": logo_churn_flag,
    })

    return df


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    path = out_dir / "saas_accounts.csv"
    df.to_csv(path, index=False)
    print(f"Generated {len(df)} accounts → {path}")
    print(f"Churn rate: {df['logo_churn_flag'].mean():.2%}")
    print(f"Plan distribution:\n{df['plan_type'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


if __name__ == "__main__":
    main()
