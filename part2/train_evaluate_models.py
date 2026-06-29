import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "part1" / "cleaned_data.csv"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def make_preprocessor() -> ColumnTransformer:
    ordinal_features = ["education", "month"]
    ordinal_categories = [
        ["missing", "primary", "secondary", "tertiary"],
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    ]
    nominal_features = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
        "day",
        "poutcome",
    ]
    numeric_features = ["age", "duration", "campaign", "pdays", "previous"]

    return ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1),
                ordinal_features,
            ),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                nominal_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def clean_feature_name(name: str) -> str:
    return name.replace("onehot__", "").replace("ordinal__", "").replace("numeric__", "")


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_
    if coefficients.ndim > 1:
        coefficients = coefficients.ravel()
    table = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in feature_names],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    return table.sort_values("abs_coefficient", ascending=False)


def regression_metrics(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    return {
        "model": name,
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def classification_metrics(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities),
    }


def threshold_sensitivity(probabilities: np.ndarray, y_test: pd.Series) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.30, 0.71, 0.10):
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_auc_difference(
    y_test: pd.Series,
    baseline_probabilities: np.ndarray,
    regularized_probabilities: np.ndarray,
    n_samples: int = 500,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    y_values = np.asarray(y_test)
    differences = []

    for _ in range(n_samples):
        indices = rng.choice(len(y_values), size=len(y_values), replace=True)
        sample_y = y_values[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        baseline_auc = roc_auc_score(sample_y, baseline_probabilities[indices])
        regularized_auc = roc_auc_score(sample_y, regularized_probabilities[indices])
        differences.append(baseline_auc - regularized_auc)

    diff_array = np.asarray(differences)
    return {
        "samples_requested": n_samples,
        "samples_used": int(len(diff_array)),
        "mean_auc_difference": float(diff_array.mean()),
        "ci_2_5_percentile": float(np.percentile(diff_array, 2.5)),
        "ci_97_5_percentile": float(np.percentile(diff_array, 97.5)),
    }


def save_roc_plot(y_test: pd.Series, probabilities: np.ndarray, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"Logistic Regression AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Term Deposit Subscription Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve_logistic_regression.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()

    print_section("1. Load cleaned_data.csv and define labels")
    df = pd.read_csv(DATA_PATH)
    df["day"] = df["day"].astype(str)
    print(f"Loaded {DATA_PATH}")
    print(f"Shape: {df.shape}")

    y_reg = df["balance"]
    y_clf = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["balance", "y"])
    print("Regression label y_reg: balance")
    print("Classification label y_clf: y == 'yes'")
    print(f"Feature matrix shape: {X.shape}")

    print_section("2-3. Encode, split, and scale without leakage")
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )
    print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}")
    print("Preprocessing and scaling are inside sklearn Pipelines, so encoders and scaler are fit on training data only.")

    print_section("4. Regression models")
    linear_model = make_model_pipeline(LinearRegression())
    ridge_model = make_model_pipeline(Ridge(alpha=1.0))
    linear_model.fit(X_train, y_reg_train)
    ridge_model.fit(X_train, y_reg_train)

    regression_comparison = pd.DataFrame(
        [
            regression_metrics("OLS Linear Regression", linear_model, X_test, y_reg_test),
            regression_metrics("Ridge Regression alpha=1.0", ridge_model, X_test, y_reg_test),
        ]
    )
    print(regression_comparison)
    regression_comparison.to_csv(REPORTS_DIR / "regression_comparison.csv", index=False)

    linear_coefficients = coefficient_table(linear_model)
    ridge_coefficients = coefficient_table(ridge_model)
    print("\nTop 10 OLS coefficients by absolute value:")
    print(linear_coefficients.head(10))
    linear_coefficients.to_csv(REPORTS_DIR / "linear_regression_coefficients.csv", index=False)
    ridge_coefficients.to_csv(REPORTS_DIR / "ridge_regression_coefficients.csv", index=False)

    print_section("5. Classification model")
    train_counts = y_clf_train.value_counts().sort_index()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_clf_train)
    weight_table = pd.DataFrame(
        {
            "class": [0, 1],
            "class_label": ["no", "yes"],
            "train_count_before_weighting": [int(train_counts.get(0, 0)), int(train_counts.get(1, 0))],
            "class_weight": class_weights,
            "effective_weighted_count": [
                train_counts.get(0, 0) * class_weights[0],
                train_counts.get(1, 0) * class_weights[1],
            ],
        }
    )
    print("Class balance before and after class_weight='balanced':")
    print(weight_table)
    weight_table.to_csv(REPORTS_DIR / "class_balance_weighting.csv", index=False)

    logistic_model = make_model_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")
    )
    logistic_model.fit(X_train, y_clf_train)
    y_clf_pred = logistic_model.predict(X_test)
    y_clf_proba = logistic_model.predict_proba(X_test)[:, 1]

    conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
    class_report_text = classification_report(y_clf_test, y_clf_pred, target_names=["no", "yes"])
    class_report_dict = classification_report(y_clf_test, y_clf_pred, target_names=["no", "yes"], output_dict=True)
    auc = roc_auc_score(y_clf_test, y_clf_proba)

    print("Confusion matrix:")
    print(conf_matrix)
    print("\nClassification report:")
    print(class_report_text)
    print(f"AUC: {auc:.4f}")
    pd.DataFrame(conf_matrix, index=["actual_no", "actual_yes"], columns=["pred_no", "pred_yes"]).to_csv(
        REPORTS_DIR / "confusion_matrix.csv"
    )
    pd.DataFrame(class_report_dict).transpose().to_csv(REPORTS_DIR / "classification_report.csv")
    save_roc_plot(y_clf_test, y_clf_proba, auc)

    print_section("5b. Decision-threshold sensitivity")
    threshold_table = threshold_sensitivity(y_clf_proba, y_clf_test)
    print(threshold_table)
    threshold_table.to_csv(REPORTS_DIR / "threshold_sensitivity.csv", index=False)

    print_section("6. Logistic regression regularization experiment")
    strong_regularized_model = make_model_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.01, solver="lbfgs")
    )
    strong_regularized_model.fit(X_train, y_clf_train)
    regularized_proba = strong_regularized_model.predict_proba(X_test)[:, 1]
    logistic_comparison = pd.DataFrame(
        [
            classification_metrics("Logistic Regression C=1.0", logistic_model, X_test, y_clf_test),
            classification_metrics("Logistic Regression C=0.01", strong_regularized_model, X_test, y_clf_test),
        ]
    )
    print(logistic_comparison)
    logistic_comparison.to_csv(REPORTS_DIR / "logistic_regularization_comparison.csv", index=False)

    print_section("7. Bootstrap confidence interval for AUC difference")
    bootstrap_summary = bootstrap_auc_difference(y_clf_test, y_clf_proba, regularized_proba)
    print(json.dumps(bootstrap_summary, indent=2))
    (REPORTS_DIR / "bootstrap_auc_difference.json").write_text(json.dumps(bootstrap_summary, indent=2))

    print_section("Save models")
    joblib.dump(linear_model, MODELS_DIR / "linear_regression_balance.joblib")
    joblib.dump(ridge_model, MODELS_DIR / "ridge_regression_balance.joblib")
    joblib.dump(logistic_model, MODELS_DIR / "logistic_regression_c1.joblib")
    joblib.dump(strong_regularized_model, MODELS_DIR / "logistic_regression_c001.joblib")
    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
