import io
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
CLEANED_PATH = BASE_DIR / "cleaned_data.csv"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_CSV_PATH = RAW_DIR / "bank-full.csv"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    if RAW_CSV_PATH.exists():
        return

    print("Downloading UCI Bank Marketing dataset...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
        bank_zip_bytes = outer_zip.read("bank.zip")

    with zipfile.ZipFile(io.BytesIO(bank_zip_bytes)) as inner_zip:
        raw_csv = inner_zip.read("bank-full.csv")

    RAW_CSV_PATH.write_bytes(raw_csv)
    print(f"Saved raw dataset to {RAW_CSV_PATH}")


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(REPORTS_DIR / f"{name}.csv")


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isnull().sum()
    percentages = (counts / df.shape[0]) * 100
    return pd.DataFrame({"missing_count": counts, "missing_percent": percentages.round(2)})


def outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append(
            {
                "column": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_rows": int(count),
                "outlier_percent": round((count / len(df)) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def strongest_corr_pair(corr: pd.DataFrame) -> tuple[str, str, float]:
    masked = corr.abs().where(~pd.DataFrame(True, corr.index, corr.columns).where(lambda x: x).values)
    pairs = []
    cols = corr.columns
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            pairs.append((left, right, corr.loc[left, right], abs(corr.loc[left, right])))
    left, right, value, _ = max(pairs, key=lambda row: row[3])
    return left, right, value


def corr_difference_table(pearson: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = pearson.columns
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            rows.append(
                {
                    "column_a": left,
                    "column_b": right,
                    "pearson": pearson.loc[left, right],
                    "spearman": spearman.loc[left, right],
                    "abs_difference": abs(abs(spearman.loc[left, right]) - abs(pearson.loc[left, right])),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_difference", ascending=False)

# Visualizations
def save_plots(df: pd.DataFrame, numeric_cols: list[str], most_skewed_col: str) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(11, 5))
    plt.plot(df.index[:500], df["balance"].head(500), linewidth=1)
    plt.title("Balance for First 500 Campaign Records")
    plt.xlabel("Row index")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_line_balance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    df.groupby("job", observed=True)["balance"].mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Mean Balance by Job")
    plt.xlabel("Job")
    plt.ylabel("Mean balance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_bar_mean_balance_by_job.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(df[most_skewed_col], bins=20, kde=True)
    plt.title(f"Histogram of Most Skewed Numeric Column: {most_skewed_col}")
    plt.xlabel(most_skewed_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_histogram_most_skewed.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df.sample(min(3000, len(df)), random_state=42), x="duration", y="campaign", alpha=0.45)
    plt.title("Duration vs Campaign Contacts")
    plt.xlabel("Duration")
    plt.ylabel("Campaign contacts")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_scatter_duration_campaign.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="y", y="duration")
    plt.title("Duration by Term Deposit Outcome")
    plt.xlabel("Subscribed to term deposit")
    plt.ylabel("Duration")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_box_duration_by_target.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson Correlation Heatmap for Numeric Columns")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_correlation_heatmap.png", dpi=150)
    plt.close()

# Main Wrokflow
def main() -> None:
    ensure_dirs()
    download_dataset()
    
    # Load the dataset
    print_section("1. Load the dataset")
    df = pd.read_csv(RAW_CSV_PATH, sep=";")
    print("First five rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print(f"\nShape: {df.shape}")

    # UCI uses the string "unknown" for missing categorical information.
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df = df.replace({"unknown": pd.NA})
    
    # Null value analysis:
    print_section("2. Null value analysis")
    null_before = missing_table(df)
    print(null_before)
    save_table(null_before, "null_percentages_before_cleaning")
    print("\nColumns exceeding 20% missing:")
    print(null_before[null_before["missing_percent"] > 20])

    numeric_cols_initial = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols_initial:
        if 0 < df[col].isnull().mean() <= 0.20:
            df[col] = df[col].fillna(df[col].median())
            
    # Duplicate detection and removal:
    print_section("3. Duplicate detection and removal")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows before removal: {duplicate_count}")
    df = df.drop_duplicates().copy()
    print(f"Rows removed: {duplicate_count}")
    null_after_duplicates = missing_table(df)
    print("\nNull percentages after duplicate removal:")
    print(null_after_duplicates)
    save_table(null_after_duplicates, "null_percentages_after_duplicates")

    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("missing")

    # Data type correction
    print_section("4. Data type correction")
    memory_before = df.memory_usage(deep=True).sum()
    print(f"Memory before type correction: {memory_before} bytes")

    df["day"] = df["day"].astype("category")
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    memory_after = df.memory_usage(deep=True).sum()
    print(f"Memory after type correction: {memory_after} bytes")
    print(f"Memory reduction: {memory_before - memory_after} bytes")
    print("\nCorrected dtypes:")
    print(df.dtypes)

    # Descriptive statistics
    print_section("5. Descriptive statistics and skewness")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    descriptive_stats = df[numeric_cols].describe()
    skewness = df[numeric_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
    print(descriptive_stats)
    print("\nSkewness:")
    print(skewness)
    save_table(descriptive_stats, "descriptive_statistics")
    save_table(skewness.to_frame("skewness"), "skewness")
    most_skewed_col = skewness.abs().idxmax()
    print(f"\nHighest absolute skewness column: {most_skewed_col} ({skewness.loc[most_skewed_col]:.4f})")
    
    # Outlier detection with IQR
    print_section("6. Outlier detection with IQR")
    outlier_cols = ["balance", "duration"]
    outliers = outlier_summary(df, outlier_cols)
    print(outliers)
    save_table(outliers.set_index("column"), "outlier_summary")
    
    # Visualizations 
    print_section("7. Visualizations")
    save_plots(df, numeric_cols, most_skewed_col)
    for plot_path in sorted(PLOTS_DIR.glob("*.png")):
        print(f"Saved {plot_path}")

    # Correlation heat map
    print_section("8. Correlation heat map")
    pearson_corr = df[numeric_cols].corr()
    print(pearson_corr)
    save_table(pearson_corr, "pearson_correlation")
    corr_a, corr_b, corr_value = strongest_corr_pair(pearson_corr)
    print(f"\nHighest absolute Pearson correlation pair: {corr_a} and {corr_b} ({corr_value:.4f})")

    # Imputation strategy comparison
    print_section("9a. Imputation strategy comparison")
    top_two_skewed = skewness.abs().head(2).index.tolist()
    imputation_rows = []
    for col in top_two_skewed:
        imputation_rows.append(
            {
                "column": col,
                "skewness": skewness.loc[col],
                "mean_before_imputation": df[col].mean(),
                "median_before_imputation": df[col].median(),
                "missing_before": int(df[col].isnull().sum()),
            }
        )
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    imputation_comparison = pd.DataFrame(imputation_rows)
    print(imputation_comparison)
    print("\nRemaining nulls for top two skewed columns:")
    print(df[top_two_skewed].isnull().sum())
    save_table(imputation_comparison.set_index("column"), "imputation_strategy_comparison")

    # Spearman rank correlation
    print_section("9b. Spearman rank correlation")
    spearman_corr = df[numeric_cols].corr(method="spearman")
    corr_diff = corr_difference_table(pearson_corr, spearman_corr)
    print("Spearman correlation matrix:")
    print(spearman_corr)
    print("\nTop three |Spearman| - |Pearson| differences:")
    print(corr_diff.head(3))
    save_table(spearman_corr, "spearman_correlation")
    save_table(corr_diff, "correlation_difference")

    # Groubed aggregation 
    print_section("9c. Grouped aggregation")
    grouped = df.groupby("job", observed=True)["balance"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(grouped)
    save_table(grouped, "grouped_balance_by_job")
    highest_mean_group = grouped["mean"].idxmax()
    highest_std_group = grouped["std"].idxmax()
    mean_ratio = grouped["mean"].max() / grouped["mean"].min()
    print(f"\nHighest mean group: {highest_mean_group}")
    print(f"Highest std group: {highest_std_group}")
    print(f"Highest-to-lowest group mean ratio: {mean_ratio:.2f}")

    # Save clean dataset
    print_section("10. Save clean dataset")
    df.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_PATH}")
    print("\nFinal null counts:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
