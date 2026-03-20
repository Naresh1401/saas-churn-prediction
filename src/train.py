"""
Model training pipeline for SaaS Account Churn Prediction.

Trains Logistic Regression, Random Forest, and XGBoost with SMOTE
and GridSearchCV, evaluates on a held-out test set, saves the best
model, and generates SHAP + health-score distribution plots.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.data_processing import clean_data, load_raw_data, split_data  # noqa: E402
from src.feature_engineering import prepare_features, compute_health_score  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Model definitions & hyper-parameter grids
# ---------------------------------------------------------------------------

MODELS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=1000, solver="liblinear"),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
        },
    },
}

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Over-sample the minority class using SMOTE."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def train_and_tune(X_train, y_train):
    """Run GridSearchCV for each model, return {name: {best_estimator, ...}}."""
    results = {}
    for name, cfg in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name} ...")
        gs = GridSearchCV(
            cfg["estimator"],
            cfg["params"],
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        results[name] = {
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_f1_cv": gs.best_score_,
        }
        print(f"  Best CV F1: {gs.best_score_:.4f}")
        print(f"  Params: {gs.best_params_}")
    return results


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def select_best_model(results: dict, X_test, y_test):
    """Pick the model with the highest test ROC-AUC."""
    best_name, best_score, best_model = None, -1, None
    all_metrics = {}

    for name, res in results.items():
        m = evaluate_model(res["best_estimator"], X_test, y_test)
        all_metrics[name] = m
        print(f"\n{name} — AUC: {m['roc_auc']}, F1: {m['f1']}, "
              f"Precision: {m['precision']}, Recall: {m['recall']}")
        if m["roc_auc"] > best_score:
            best_score = m["roc_auc"]
            best_name = name
            best_model = res["best_estimator"]

    return best_name, best_model, all_metrics


def save_model(model, scaler, feature_names: list[str]):
    """Persist the model, scaler, and feature list."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    joblib.dump(artifact, MODELS_DIR / "best_model.pkl")
    print(f"\nModel saved → {MODELS_DIR / 'best_model.pkl'}")


def generate_shap_plot(model, X_test: pd.DataFrame):
    """Create and save a SHAP summary plot."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved → {OUTPUTS_DIR / 'shap_summary.png'}")


def generate_health_score_risk_plot(raw_df: pd.DataFrame, model, X_test, y_test):
    """Health score distribution coloured by predicted risk tier."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute health scores for test set from raw data aligned by index
    clean_subset = raw_df.loc[X_test.index].copy()
    health_scores = compute_health_score(clean_subset)

    risk_tier = pd.cut(
        y_prob,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    for tier in ["Low", "Medium", "High"]:
        mask = risk_tier == tier
        ax.hist(
            health_scores[mask],
            bins=30,
            alpha=0.6,
            label=f"{tier} Risk (n={mask.sum()})",
            color=colours[tier],
            edgecolor="white",
        )
    ax.set_xlabel("Health Score")
    ax.set_ylabel("Count")
    ax.set_title("Account Health Score Distribution by Predicted Risk Tier")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "health_score_risk.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Health score plot saved → {OUTPUTS_DIR / 'health_score_risk.png'}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("SAAS ACCOUNT CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load & clean
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    print(f"\nDataset: {clean_df.shape[0]} accounts, churn rate = "
          f"{clean_df['logo_churn_flag'].mean():.2%}")

    # 2. Split BEFORE feature engineering (avoid leakage)
    X_train_raw, X_test_raw, y_train, y_test = split_data(clean_df)

    # 3. Feature engineering
    X_train, scaler = prepare_features(X_train_raw, fit=True)
    X_test, _ = prepare_features(X_test_raw, scaler=scaler, fit=False)

    feature_names = list(X_train.columns)

    # 4. SMOTE on training data
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

    # 5. Train & tune
    results = train_and_tune(X_train_sm, y_train_sm)

    # 6. Evaluate & select best
    best_name, best_model, all_metrics = select_best_model(results, X_test, y_test)
    print(f"\n*** Best model: {best_name} ***")

    # 7. Save artefacts
    save_model(best_model, scaler, feature_names)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 8. SHAP plot
    generate_shap_plot(best_model, X_test)

    # 9. Health score distribution by risk tier
    generate_health_score_risk_plot(clean_df, best_model, X_test, y_test)

    print("\n✓ Training pipeline complete.")


if __name__ == "__main__":
    main()
