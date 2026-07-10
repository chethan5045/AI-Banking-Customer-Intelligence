# Part 2 - Supervised Machine Learning Model

## Overview

In this part of the project, I used the cleaned dataset created in Part 1 to build and evaluate two supervised machine learning models.
The first model predicts a customer's account balance using a regression approach, while the second predicts whether a customer is likely to subscribe to a term deposit using binary classification.
To ensure fair model training, the feature set excludes both the balance and y columns, preventing target leakage.

## Goal

This part uses `part1/cleaned_data.csv` to build and evaluate:

- A regression model that predicts customer `balance`
- A binary classification model that predicts whether `y == "yes"` for term-deposit subscription

The shared feature matrix `X` uses all columns except `balance` and `y`. This avoids leaking either target into the feature set.

## Running the project

From the repository root:

```bash
pip install -r requirements.txt
python3 part2/train_evaluate_models.py
```

The dataset creates:

- `part2/plots/roc_curve_logistic_regression.png`
- metric CSV files in `part2/reports/`
- trained `.joblib` model pipelines in `part2/models/`

## Labels Target Variables

Regression label:

```python
y_reg = df["balance"]
```

Classification label:

```python
y_clf = (df["y"] == "yes").astype(int)
```

`y_clf = 1` means the customer subscribed to a term deposit. `y_clf = 0` means the customer did not subscribe.

## Encoding and Scaling

The dataset uses a scikit-learn `Pipeline` with a `ColumnTransformer`.

Ordinal encoding is used for:

- `education`: `missing < primary < secondary < tertiary`
- `month`: `jan < feb < ... < dec`

These columns have a natural order. For unordered categorical columns such as `job`, `marital`, `contact`, `poutcome`, and `day`, the dataset uses one-hot encoding with `drop="first"` to reduce multicollinearity. One-hot encoding avoids creating a false ordinal relationship, for example pretending that one job category is numerically greater than another.

`StandardScaler` is fit only inside the training pipeline. Fitting the scaler on the full dataset would be data leakage because the scaler would learn test-set means and standard deviations before model evaluation.

## Regression Results

| Model | MSE | R2 |
| --- | ---: | ---: |
| OLS Linear Regression | 8,303,771.97 | 0.0443 |
| Ridge Regression alpha=1.0 | 8,303,759.38 | 0.0443 |

The R2 score is low, so the available campaign and profile features explain only a small part of account-balance variation. This is reasonable because customer balance depends on many financial variables not present in this dataset.

Top OLS coefficients by absolute value:

| Feature | Coefficient |
| --- | ---: |
| `age` | 300.89 |
| `loan_yes` | -219.70 |
| `month` | 209.43 |

Because features are scaled, a large positive coefficient means a one-standard-deviation increase in that feature is associated with a larger predicted balance. A large negative coefficient means a one-standard-deviation increase is associated with a lower predicted balance. For example, `loan_yes` has a negative coefficient, so customers with personal loans are predicted to have lower balances after controlling for the other encoded features.

Ridge regression gives almost identical MSE and R2 here. Ridge adds an L2 penalty that shrinks coefficients. Its `alpha` parameter controls penalty strength: larger `alpha` means stronger shrinkage. Ridge can produce a different coefficient profile from OLS when features are correlated or when many encoded dummy variables are present.

## Classification and Class Imbalance

Training class distribution before weighting:

| Class | Label | Count | Class weight | Effective weighted count |
| --- | --- | ---: | ---: | ---: |
| 0 | no | 31,937 | 0.5662 | 18,084 |
| 1 | yes | 4,231 | 4.2742 | 18,084 |

The positive class is only about 11.7% of the dataset, so it is imbalanced. I used `class_weight="balanced"` in logistic regression. This keeps the original rows but gives the minority class more loss weight during training.

Confusion matrix:

|  | Predicted no | Predicted yes |
| --- | ---: | ---: |
| Actual no | 6,667 | 1,318 |
| Actual yes | 221 | 837 |

Classification metrics for `LogisticRegression(C=1.0, class_weight="balanced")`:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8298 |
| Precision for yes | 0.3884 |
| Recall for yes | 0.7911 |
| F1 for yes | 0.5210 |
| AUC | 0.8947 |

Precision formula:

```text
Precision = TP / (TP + FP)
```

Recall formula:

```text
Recall = TP / (TP + FN)
```

For this banking marketing task, recall is especially important if the business wants to avoid missing customers who are likely to subscribe. A false negative means a potentially valuable lead is missed. The cost is that higher recall usually creates more false positives, meaning more customers may be contacted unnecessarily.

The AUC value of `0.8947` means the model has strong ability to rank actual subscribers above non-subscribers across decision thresholds.

## Decision Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0.30 | 0.2715 | 0.9263 | 0.4200 |
| 0.40 | 0.3336 | 0.8696 | 0.4822 |
| 0.50 | 0.3884 | 0.7911 | 0.5210 |
| 0.60 | 0.4471 | 0.7108 | 0.5489 |
| 0.70 | 0.5149 | 0.6191 | 0.5622 |

The best F1 score among these tested thresholds is at `0.70`. If the bank wants the best balance between precision and recall, I would raise the threshold from `0.50` to `0.70`. If the bank cares more about finding as many likely subscribers as possible, I would lower the threshold toward `0.30` or `0.40`, accepting more false positives.

## Regularization Experiment

| Model | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression C=1.0 | 0.8298 | 0.3884 | 0.7911 | 0.5210 | 0.8947 |
| Logistic Regression C=0.01 | 0.8299 | 0.3886 | 0.7911 | 0.5212 | 0.8947 |

In logistic regression, `C` controls inverse regularization strength. Smaller `C` means stronger L2 regularization. Reducing `C` to `0.01` produced almost the same performance on this dataset, with a tiny F1 increase and tiny AUC decrease. The difference is not meaningful in practical terms.

## Bootstrap AUC Difference

The dataset drew 500 bootstrap samples from the test set and computed:

```text
AUC difference = AUC(C=1.0) - AUC(C=0.01)
```

Results:

- Mean AUC difference: `0.0000076`
- 95% CI lower bound: `-0.0003175`
- 95% CI upper bound: `0.0003237`

The confidence interval includes zero, so the AUC difference is not reliable. There is no evidence that `C=1.0` consistently outperforms `C=0.01` on this test set.

## Project Outputs

- `train_evaluate_models.py`: full preprocessing, training, and evaluation dataset
- `plots/roc_curve_logistic_regression.png`: ROC curve with AUC annotation
- `reports/`: regression, classification, threshold, coefficient, and bootstrap results
- `models/`: saved trained pipelines
