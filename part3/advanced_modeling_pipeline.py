import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = ROOT_DIR / "part1" / "cleaned_data.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = BASE_DIR / "best_model.pkl"


def ensure_dirs() -> None:
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


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def clean_feature_name(name: str) -> str:
    return name.replace("onehot__", "").replace("ordinal__", "").replace("numeric__", "")


def evaluate_classifier(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        test_score = model.predict_proba(X_test)[:, 1]
    else:
        test_score = model.decision_function(X_test)
    return {
        "model": name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_auc": roc_auc_score(y_test, test_score),
    }


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(REPORTS_DIR / f"{name}.csv", index=False)


def main() -> None:
    ensure_dirs()

    print_section("Load data and split")
    df = pd.read_csv(DATA_PATH)
    df["day"] = df["day"].astype(str)
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["balance", "y"])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"X_train: {X_train.shape}; X_test: {X_test.shape}")

    preprocessor = make_preprocessor()
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    print(f"Encoded/scaled feature count: {len(feature_names)}")

    print_section("1. Decision Tree baseline")
    default_tree = DecisionTreeClassifier(random_state=42)
    default_tree.fit(X_train_scaled, y_train)
    default_tree_metrics = evaluate_classifier(
        "Decision Tree default", default_tree, X_train_scaled, X_test_scaled, y_train, y_test
    )
    print(default_tree_metrics)

    print_section("2. Controlled Decision Tree")
    controlled_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    controlled_tree.fit(X_train_scaled, y_train)
    controlled_tree_metrics = evaluate_classifier(
        "Decision Tree max_depth=5 min_samples_split=20",
        controlled_tree,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )
    print(controlled_tree_metrics)

    print_section("3. Gini vs Entropy comparison")
    gini_tree = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=42)
    entropy_tree = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)
    gini_tree.fit(X_train_scaled, y_train)
    entropy_tree.fit(X_train_scaled, y_train)
    criterion_comparison = pd.DataFrame(
        [
            evaluate_classifier("Decision Tree gini max_depth=5", gini_tree, X_train_scaled, X_test_scaled, y_train, y_test),
            evaluate_classifier(
                "Decision Tree entropy max_depth=5",
                entropy_tree,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
            ),
        ]
    )
    print(criterion_comparison)
    save_table(criterion_comparison, "gini_entropy_comparison")

    print_section("4. Random Forest")
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    random_forest.fit(X_train_scaled, y_train)
    rf_metrics = evaluate_classifier("Random Forest", random_forest, X_train_scaled, X_test_scaled, y_train, y_test)
    print(rf_metrics)
    importance_table = pd.DataFrame(
        {"feature": feature_names, "importance": random_forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nTop 5 feature importances:")
    print(importance_table.head(5))
    print("\nLowest 5 feature importances:")
    print(importance_table.tail(5))
    save_table(importance_table, "random_forest_feature_importances")

    print_section("4a. Gradient Boosting")
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gradient_boosting.fit(X_train_scaled, y_train)
    gb_metrics = evaluate_classifier(
        "Gradient Boosting",
        gradient_boosting,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )
    print(gb_metrics)

    print_section("4b. Feature ablation study")
    lowest_features = importance_table.tail(5)["feature"].tolist()
    keep_indices = [i for i, name in enumerate(feature_names) if name not in set(lowest_features)]
    reduced_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    reduced_rf.fit(X_train_scaled[:, keep_indices], y_train)
    full_rf_auc = roc_auc_score(y_test, random_forest.predict_proba(X_test_scaled)[:, 1])
    reduced_rf_auc = roc_auc_score(y_test, reduced_rf.predict_proba(X_test_scaled[:, keep_indices])[:, 1])
    ablation_table = pd.DataFrame(
        [
            {
                "model": "Random Forest full features",
                "removed_features": "",
                "feature_count": len(feature_names),
                "test_auc": full_rf_auc,
            },
            {
                "model": "Random Forest reduced features",
                "removed_features": ", ".join(lowest_features),
                "feature_count": len(keep_indices),
                "test_auc": reduced_rf_auc,
            },
        ]
    )
    print(ablation_table)
    save_table(ablation_table, "feature_ablation")

    print_section("5. Cross-validated comparison")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_models = {
        "Logistic Regression C=1.0": make_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        ),
        "Controlled Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
        ),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)),
        "Gradient Boosting": make_pipeline(
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        ),
    }
    cv_rows = []
    for name, pipeline in cv_models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_rows.append({"model": name, "cv_mean_auc": scores.mean(), "cv_std_auc": scores.std()})
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    cv_comparison = pd.DataFrame(cv_rows)
    save_table(cv_comparison, "cross_validated_comparison")

    print_section("6. Hyperparameter tuning with GridSearchCV")
    rf_pipeline = make_pipeline(RandomForestClassifier(random_state=42, n_jobs=1))
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [1, 5],
    }
    total_configurations = (
        len(param_grid["model__n_estimators"])
        * len(param_grid["model__max_depth"])
        * len(param_grid["model__min_samples_leaf"])
    )
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    grid_summary = {
        "best_params": grid_search.best_params_,
        "best_cv_auc": grid_search.best_score_,
        "total_configurations": total_configurations,
        "total_model_fits": total_configurations * cv.get_n_splits(),
    }
    print(json.dumps(grid_summary, indent=2))
    (REPORTS_DIR / "grid_search_summary.json").write_text(json.dumps(grid_summary, indent=2))
    pd.DataFrame(grid_search.cv_results_).to_csv(REPORTS_DIR / "grid_search_cv_results.csv", index=False)

    print_section("7. Manual learning curve")
    learning_rows = []
    for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]:
        row_count = int(fraction * len(X_train))
        subset_X = X_train.iloc[:row_count]
        subset_y = y_train.iloc[:row_count]
        model = clone(best_pipeline)
        model.fit(subset_X, subset_y)
        train_auc = roc_auc_score(subset_y, model.predict_proba(subset_X)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        learning_rows.append(
            {
                "training_fraction": fraction,
                "training_rows": row_count,
                "training_auc": train_auc,
                "test_auc": test_auc,
            }
        )
    learning_curve = pd.DataFrame(learning_rows)
    print(learning_curve)
    save_table(learning_curve, "manual_learning_curve")

    print_section("8. Serialize and reload best model")
    joblib.dump(best_pipeline, BEST_MODEL_PATH, compress=3)
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl", compress=3)
    reloaded_model = joblib.load(BEST_MODEL_PATH)
    handcrafted_rows = pd.DataFrame(
        [
            {
                "age": 35,
                "job": "management",
                "marital": "single",
                "education": "tertiary",
                "default": "no",
                "housing": "no",
                "loan": "no",
                "contact": "cellular",
                "day": "15",
                "month": "may",
                "duration": 320,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "missing",
            },
            {
                "age": 52,
                "job": "blue-collar",
                "marital": "married",
                "education": "secondary",
                "default": "no",
                "housing": "yes",
                "loan": "yes",
                "contact": "missing",
                "day": "5",
                "month": "may",
                "duration": 80,
                "campaign": 4,
                "pdays": -1,
                "previous": 0,
                "poutcome": "missing",
            },
        ]
    )
    reload_predictions = reloaded_model.predict(handcrafted_rows)
    reload_probabilities = reloaded_model.predict_proba(handcrafted_rows)[:, 1]
    reload_results = handcrafted_rows.copy()
    reload_results["predicted_class"] = reload_predictions
    reload_results["yes_probability"] = reload_probabilities
    print(reload_results[["predicted_class", "yes_probability"]])
    save_table(reload_results, "reload_predict_results")

    print_section("9. Summary comparison table")
    test_auc_rows = [
        {
            "model": "Logistic Regression C=1.0",
            "test_auc": 0.8947059290044069,
        },
        {
            "model": "Controlled Decision Tree",
            "test_auc": controlled_tree_metrics["test_auc"],
        },
        {
            "model": "Random Forest",
            "test_auc": rf_metrics["test_auc"],
        },
        {
            "model": "Gradient Boosting",
            "test_auc": gb_metrics["test_auc"],
        },
        {
            "model": "Tuned Random Forest",
            "test_auc": roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1]),
        },
    ]
    test_auc_table = pd.DataFrame(test_auc_rows)
    summary_comparison = cv_comparison.merge(test_auc_table, on="model", how="outer")
    summary_comparison.loc[
        summary_comparison["model"] == "Tuned Random Forest", ["cv_mean_auc", "cv_std_auc"]
    ] = [grid_search.best_score_, np.nan]
    print(summary_comparison)
    save_table(summary_comparison, "summary_comparison")


if __name__ == "__main__":
    main()
