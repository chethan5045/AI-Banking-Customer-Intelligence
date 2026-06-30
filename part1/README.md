# Part 1 - Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset

This part uses the public UCI Bank Marketing dataset.

- Source: UCI Machine Learning Repository, Bank Marketing dataset
- Download URL used by the script: `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip`
- File used: `bank-full.csv`
- Original shape: 45,211 rows and 17 columns
- Target for later parts: `y`, whether a customer subscribed to a term deposit

The script downloads the raw dataset into `part1/data/raw/`. That raw folder is ignored by Git because the script documents and reproduces the download. The cleaned dataset is saved as `part1/cleaned_data.csv`.

## How to Run

From the repository root:

```bash
pip install -r requirements.txt
python3 part1/eda_bank_marketing.py
```

The script prints the required analysis and creates:

- `part1/cleaned_data.csv`
- `part1/plots/01_line_balance.png`
- `part1/plots/02_bar_mean_balance_by_job.png`
- `part1/plots/03_histogram_most_skewed.png`
- `part1/plots/04_scatter_duration_campaign.png`
- `part1/plots/05_box_duration_by_target.png`
- `part1/plots/06_correlation_heatmap.png`
- CSV report tables in `part1/reports/`

## Loading and Initial Inspection

The data is loaded with:

```python
pd.read_csv(RAW_CSV_PATH, sep=";")
```

The script prints:

- First five rows
- Column data types
- DataFrame shape

The original shape is `(45211, 17)`.

## Null Value Analysis

The UCI dataset uses the string `unknown` for missing categorical information. The script converts `unknown` to null values for reporting.

Columns above 20% missing:

| Column | Missing count | Missing percent |
| --- | ---: | ---: |
| `contact` | 13,020 | 28.80% |
| `poutcome` | 36,959 | 81.75% |

No numeric columns had missing values. For numeric imputation, median is preferred over mean because several numeric variables are strongly skewed. The mean would be pulled toward extreme values, while the median better represents a typical customer.

For the final cleaned dataset, missing categorical values are filled with the explicit label `missing`. This preserves missingness as information and leaves `cleaned_data.csv` with zero null values.

## Duplicate Detection

Duplicate rows found:

```text
0
```

No rows were removed, so duplicate removal did not change any null percentages.

## Data Type Correction

The `day` column is read as an integer, but it represents a day-of-month label rather than a continuous measurement. It was converted to `category`.

Repeated string columns such as `job`, `marital`, `education`, `contact`, `month`, `poutcome`, and `y` were also converted to `category`.

Memory usage:

| Stage | Memory |
| --- | ---: |
| Before conversion | 8,462,925 bytes |
| After conversion | 2,668,477 bytes |
| Reduction | 5,794,448 bytes |

## Descriptive Statistics and Skewness

Numeric columns analyzed:

```text
age, balance, duration, campaign, pdays, previous
```

Highest absolute skewness:

```text
previous = 41.8465
```

`previous` is positively skewed. Most customers have zero previous campaign contacts, while a small number have very high values. In a positively skewed distribution, the mean is pulled upward by extreme high values. This makes the median a safer imputation statistic than the mean.

Top two skewed columns:

| Column | Skewness | Mean | Median |
| --- | ---: | ---: | ---: |
| `previous` | 41.8465 | 0.5803 | 0.0 |
| `balance` | 8.3603 | 1362.2721 | 448.0 |

Both columns had zero missing values, and `isnull().sum()` confirmed no nulls remained.

## IQR Outlier Detection

Outliers were counted but not removed.

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outlier rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balance` | 72.0 | 1428.0 | 1356.0 | -1962.0 | 3462.0 | 4,729 |
| `duration` | 103.0 | 319.0 | 216.0 | -221.0 | 643.0 | 3,235 |

I will retain these outliers for Part 2 because high balances and long call durations may contain real predictive signal. Dropping them could remove important customer behavior.

## Required Visualizations

The script saves all required plots using `plt.savefig()`.

| Plot | File | Interpretation |
| --- | --- | --- |
| Line plot | `plots/01_line_balance.png` | Balance varies sharply across records, with visible spikes. |
| Bar chart | `plots/02_bar_mean_balance_by_job.png` | Retired customers have the highest mean balance. |
| Histogram | `plots/03_histogram_most_skewed.png` | `previous` is concentrated at zero with a long right tail. |
| Scatter plot | `plots/04_scatter_duration_campaign.png` | `duration` and `campaign` have a weak relationship. |
| Box plot | `plots/05_box_duration_by_target.png` | Subscribers tend to have higher call duration. |
| Heat map | `plots/06_correlation_heatmap.png` | Most numeric correlations are weak. |

## Correlation Heat Map

The strongest absolute Pearson correlation is:

```text
pdays and previous = 0.4548
```

This does not prove causation. A plausible alternative explanation is prior campaign engagement: customers contacted in earlier campaigns naturally have both a non-default `pdays` value and a higher `previous` count.

## Spearman Rank Correlation

Largest differences between Spearman and Pearson:

| Pair | Pearson | Spearman | Difference |
| --- | ---: | ---: | ---: |
| `pdays` / `previous` | 0.4548 | 0.9856 | 0.5308 |
| `campaign` / `previous` | -0.0329 | -0.1084 | 0.0756 |
| `balance` / `pdays` | 0.0034 | 0.0697 | 0.0662 |

For `pdays` and `previous`, Spearman is much stronger, suggesting a monotonic but non-linear relationship. For Part 2 feature guidance, both Pearson and Spearman are useful, but Spearman is especially helpful for skewed campaign-history variables.

## Grouped Aggregation

Grouped aggregation was computed with:

```python
df.groupby("job")["balance"].agg(["mean", "std", "count"])
```

Findings:

- Highest mean balance group: `retired`, mean `1984.22`
- Highest standard deviation group: `retired`, std `4397.04`
- Highest-to-lowest group mean ratio: `1.99`

High within-group standard deviation means job alone is not enough to predict balance reliably for every customer. The nearly 2x mean ratio still suggests `job` carries useful predictive signal for later modeling.

## Final Output

The final cleaned dataset is saved as:

```text
part1/cleaned_data.csv
```

The final dataset has zero null values and is used in Parts 2 and 3.
