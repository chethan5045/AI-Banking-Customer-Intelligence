# Part 3 - Advanced Modeling, Ensembles, Tuning, and Full ML Pipeline

## Goal

This part extends the Part 2 classification work. The target remains:

```python
y = (df["y"] == "yes").astype(int)
```

The task is to compare decision trees, ensemble models, cross-validation, hyperparameter tuning, feature ablation, a manual learning curve, and a serialized production-style sklearn pipeline.

## How to Run

From the repository root:

```bash
pip install -r requirements.txt
python3 part3/advanced_modeling_pipeline.py
```

The script creates:

- `part3/best_model.pkl`
- `part3/models/best_model.pkl`
- result files in `part3/reports/`

`best_model.pkl` is compressed with `joblib.dump(..., compress=3)` so it stays below GitHub's 100 MB file limit.

## Preprocessing Pipeline

The model pipeline uses:

- `ColumnTransformer`
- `OrdinalEncoder` for ordered columns: `education`, `month`
- `OneHotEncoder(drop="first")` for unordered categorical columns
- `SimpleImputer(strategy="median")`
- `StandardScaler`
- classifier model

Preprocessing is fit only on the training data inside sklearn pipelines to avoid test-set leakage.

## Decision Tree Baseline

| Model | Train accuracy | Test accuracy | Test AUC |
| --- | ---: | ---: | ---: |
| Unconstrained Decision Tree | 1.0000 | 0.8781 | 0.7063 |
| Controlled Decision Tree | 0.9043 | 0.8980 | 0.8667 |

The unconstrained tree shows clear overfitting: training accuracy is perfect, but test accuracy and AUC are much lower. Decision trees are high-variance models because they greedily fit splits to the training data and do not revisit earlier split decisions.

The controlled tree uses `max_depth=5` and `min_samples_split=20`. `max_depth` limits how deep the tree can grow, reducing variance at the cost of more bias. `min_samples_split` prevents splitting tiny nodes, which helps avoid noisy rules. The train/test gap is much smaller than the unconstrained tree.

## Gini vs Entropy

| Model | Train accuracy | Test accuracy | Test AUC |
| --- | ---: | ---: | ---: |
| Decision Tree gini max_depth=5 | 0.9043 | 0.8980 | 0.8667 |
| Decision Tree entropy max_depth=5 | 0.9025 | 0.8994 | 0.8792 |

Gini impurity:

```text
Gini = 1 - sum(p_i^2)
```

Entropy:

```text
Entropy = -sum(p_i * log2(p_i))
```

`p_i` is the proportion of samples from class `i` in a node. A node with Gini `0` is pure, meaning all samples in that node belong to one class.

## Random Forest

Random Forest settings:

```python
RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
```

Results:

| Train accuracy | Test accuracy | Test AUC |
| ---: | ---: | ---: |
| 0.9025 | 0.8968 | 0.9155 |

Top 5 feature importances:

| Feature | Importance |
| --- | ---: |
| `duration` | 0.3841 |
| `poutcome_success` | 0.1636 |
| `age` | 0.0608 |
| `month` | 0.0581 |
| `pdays` | 0.0565 |

Random Forest feature importance measures average reduction in Gini impurity across all trees and splits that use a feature. This differs from a linear regression coefficient: importance is not a signed one-unit effect, and it can capture non-linear split usefulness.

Bagging means each tree is trained on a bootstrap sample of the training data, and each split considers only a random subset of features. Averaging many noisy trees reduces variance compared with one deep decision tree.

## Gradient Boosting

Gradient Boosting settings:

```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)
```

Results:

| Train accuracy | Test accuracy | Test AUC |
| ---: | ---: | ---: |
| 0.9104 | 0.9057 | 0.9218 |

Gradient Boosting performed better than the untuned Random Forest on test AUC.

## Feature Ablation

The five lowest-importance Random Forest features were removed:

```text
job_housemaid, job_entrepreneur, day_31, default_yes, job_missing
```

| Model | Feature count | Test AUC |
| --- | ---: | ---: |
| Random Forest full features | 58 | 0.9155 |
| Random Forest reduced features | 53 | 0.9169 |

The reduced model slightly improved AUC, so these removed features were likely uninformative or noisy for this Random Forest setup. In production, a lower-dimensional model can reduce inference cost and maintenance burden. The trade-off is that simplification is only acceptable when AUC degradation is below the business tolerance; here there was no degradation.

## Cross-Validated Comparison

5-fold stratified CV used `scoring="roc_auc"`.

| Model | 5-fold CV mean AUC | 5-fold CV std AUC | Test AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression C=1.0 | 0.8957 | 0.0038 | 0.8947 |
| Controlled Decision Tree | 0.8571 | 0.0083 | 0.8667 |
| Random Forest | 0.9155 | 0.0027 | 0.9155 |
| Gradient Boosting | 0.9215 | 0.0013 | 0.9218 |
| Tuned Random Forest | 0.9258 | N/A | 0.9260 |

Cross-validation is more reliable than a single train-test split because it evaluates the model across multiple train/validation partitions. This reduces dependence on one lucky or unlucky split.

## GridSearchCV

Random Forest grid:

```python
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_leaf": [1, 5],
}
```

Total configurations: `18`.

Total model fits: `90` because 18 configurations were evaluated across 5 folds.

Best parameters:

```text
model__max_depth = None
model__min_samples_leaf = 1
model__n_estimators = 200
```

Best CV AUC: `0.9258`.

Exhaustive grid search tries every parameter combination, which is thorough but can be expensive. Randomized search samples from a larger parameter space and is often faster when the search space is large.

## Manual Learning Curve

| Training fraction | Training rows | Training AUC | Test AUC |
| ---: | ---: | ---: | ---: |
| 0.20 | 7,233 | 1.0000 | 0.9184 |
| 0.40 | 14,467 | 1.0000 | 0.9221 |
| 0.60 | 21,700 | 1.0000 | 0.9247 |
| 0.80 | 28,934 | 1.0000 | 0.9244 |
| 1.00 | 36,168 | 1.0000 | 0.9260 |

Training AUC stays at 1.0, which indicates the tuned Random Forest can memorize the training subsets. Test AUC generally rises as more data is used, so more training data would likely help. The model is not fully capacity-limited yet; the trend suggests some remaining data limitation.

## Reload and Predict

The script verifies that the saved model can be loaded and used for prediction:

```python
import joblib
import pandas as pd

model = joblib.load("part3/best_model.pkl")
sample_rows = pd.DataFrame([...])
predictions = model.predict(sample_rows)
probabilities = model.predict_proba(sample_rows)[:, 1]
print(predictions, probabilities)
```

This block runs successfully inside `advanced_modeling_pipeline.py` and writes results to `reports/reload_predict_results.csv`.

## Recommendation

I recommend the tuned Random Forest. It has the best CV AUC (`0.9258`) and the best test AUC (`0.9260`) among the evaluated models. Gradient Boosting is close and simpler in file size, but the tuned Random Forest gives the strongest ranking performance for identifying likely term-deposit subscribers. Because the model is serialized as a full sklearn pipeline, the same preprocessing used during training will be applied during inference.

## Files Produced

- `advanced_modeling_pipeline.py`: full Part 3 script
- `best_model.pkl`: serialized best pipeline
- `models/best_model.pkl`: duplicate saved model copy
- `reports/`: CV, grid search, feature importance, ablation, learning curve, reload prediction, and summary tables
