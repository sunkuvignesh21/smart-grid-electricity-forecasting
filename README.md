# Smart Grid Electricity Consumption Analysis and Forecasting

A Streamlit dashboard and reproducible notebook project for analysing historical household electricity use and forecasting daily mean active power (kW).

> **Scope:** the UCI dataset represents one household in Sceaux, France. This is a household-level academic forecasting prototype, not a live utility-grid monitoring or control system.

## Website features

- Download the official UCI dataset automatically, or upload the original TXT / ZIP file.
- Parse date/time values, convert measurements to numeric data, and handle missing values using time interpolation followed by forward/backward fill.
- Explore daily, hourly, weekly and monthly consumption patterns and a correlation heatmap.
- Inspect the Global Active Power distribution with a histogram, Normal Q-Q plot, Shapiro-Wilk and D'Agostino tests, skewness, kurtosis and IQR outlier checks.
- Compare original, z-score-standardised and min-max-normalised data.
- Train and compare Linear Regression and Random Forest using a chronological 80/20 split.
- Review MAE, RMSE and R-squared, plus actual-versus-predicted charts.
- Generate recursive 7–30 day daily forecasts and download forecast, metrics and prediction CSV files.

## Repository structure

```text
.
├── app.py
├── src/
│   └── pipeline.py
├── notebooks/
│   ├── Smart_Grid_Project.ipynb
│   └── review3_reproducible_analysis.ipynb
├── tests/
│   └── test_pipeline.py
├── .streamlit/
│   └── config.toml
├── .github/
│   └── workflows/
│       └── python-tests.yml
├── requirements.txt
├── runtime.txt
├── LICENSE
└── README.md
```

The original project notebook is included with its code cells preserved. Its saved cell outputs are cleared in the repository copy so the notebook can be rerun reproducibly; your uploaded local notebook is not changed.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown by Streamlit (usually http://localhost:8501). Internet access is needed for automatic dataset download; alternatively, upload the dataset file in the app sidebar.

## Run the notebooks

Open either notebook under `notebooks/` in Google Colab or Jupyter and run its cells from top to bottom. The reproducible Review 3 notebook generates normality, outlier and scaling diagnostics in addition to the EDA and forecasting workflow.

## Deploy the website

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Choose **Create app**.
3. Select repository `sunkuvignesh21/smart-grid-electricity-forecasting`, branch `main`, and main file path `app.py`.
4. Deploy and open the generated `*.streamlit.app` URL.

GitHub Pages cannot run this Python app; use Streamlit Community Cloud or another Python-capable host. The repository contains no raw 126 MB dataset file: the website downloads the official UCI archive on demand.

## Method

1. Read the UCI semicolon-separated data, treating `?` as missing.
2. Combine `Date` and `Time` into a datetime index; convert seven electrical measurements to numeric values.
3. Interpolate missing readings over time, then apply forward/backward fill for any remaining gaps.
4. Aggregate `Global_active_power` into a daily mean target.
5. Create `Lag_1`, `Lag_7`, `Lag_14`, prior 7-day rolling mean and calendar predictors.
6. Preserve time order: the first 80% of supervised daily records are used for training; the final 20% are held out for testing.
7. Use a training-only StandardScaler pipeline for Linear Regression. Random Forest is not scaled because tree splits do not depend on feature magnitude.
8. Compare both models with MAE, RMSE and R-squared. Refit the lower-RMSE model on all available supervised rows for the future forecast.

IQR flags are statistical candidates, not proof of faulty readings. Genuine electricity peaks are retained rather than automatically deleted. Normality tests assess the observed power distribution; normally distributed predictors are not required for Random Forest.

## Dataset citation

Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository, dataset 235. https://doi.org/10.24432/C58K54. Dataset licensed under CC BY 4.0.

Official page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Limitations

- One-household historical dataset; results cannot be generalised directly to a regional or national smart grid.
- Source data ends in November 2010 and has no live smart-meter feed.
- Weather, tariff, occupancy and event features are not present in the source dataset.
- Recursive forecasts are point estimates; prediction intervals are not calculated.
- This baseline is intended for academic demonstration, not operational grid decisions.

## License

Project code is released under the MIT License. The UCI dataset retains its own CC BY 4.0 license.
