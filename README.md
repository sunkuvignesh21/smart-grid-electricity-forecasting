# Smart Grid Electricity Consumption Analysis and Forecasting

A Streamlit web dashboard and reproducible notebook for analysing household electricity patterns and forecasting the next several days of **daily mean active power (kW)**.

> **Project scope:** the supplied UCI dataset represents one household in Sceaux, France. This is a historical household-level forecasting prototype, not a live utility-grid monitoring system.

## Live app

Deploy this repository using [Streamlit Community Cloud](https://share.streamlit.io/). Set the app entry point to `app.py`. The app downloads the official UCI ZIP archive on first use; alternatively, upload the original TXT or ZIP file in the sidebar. The raw dataset is intentionally not committed to GitHub.

## Features

- Automatic UCI dataset retrieval or local TXT/ZIP upload.
- Date-time parsing, numeric conversion, missing-value handling using time interpolation and forward/backward fill.
- Daily target construction and exploratory daily, hourly, weekly and monthly views.
- Correlation heatmap for the seven electrical measurement columns.
- Normality diagnostics: histogram, normal Q–Q plot, Shapiro–Wilk and D’Agostino tests, skewness and excess kurtosis.
- IQR outlier detection and comparison of original, z-score-standardised and min–max-normalised values.
- Chronological 80/20 evaluation of StandardScaler + Linear Regression and Random Forest.
- MAE, RMSE and R² metrics, actual-versus-predicted plot and downloadable CSV outputs.
- Recursive 7–30 day forecasting with the model that has lower holdout RMSE.

## Repository layout

```text
Smart_Grid_Web_Project/
├── app.py                         # Streamlit website
├── src/pipeline.py                # Data preparation, model and forecast functions
├── notebooks/
│   └── review3_reproducible_analysis.ipynb
├── tests/test_pipeline.py         # Lightweight pipeline tests
├── .streamlit/config.toml          # Streamlit theme and server settings
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The app needs internet access to download the official UCI archive unless you upload the dataset file yourself.

## Run the notebook

Open `notebooks/review3_reproducible_analysis.ipynb` in Google Colab or Jupyter and run the cells from top to bottom. The notebook downloads the official dataset archive and generates the EDA, diagnostics, scaling comparison, model metrics and `daily_electricity_forecasting.csv`.

## Deploy from GitHub to Streamlit Community Cloud

1. Create a **new, separate, public** GitHub repository named `smart-grid-electricity-forecasting`. Do not use an unrelated project repository.
2. Extract this ZIP locally, open a terminal inside the `Smart_Grid_Web_Project` folder, and run the commands below (replace the owner placeholder with your GitHub username):

   ```bash
   git init
   git add .
   git commit -m "Initial Smart Grid forecasting dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/smart-grid-electricity-forecasting.git
   git push -u origin main
   ```

   Do not add the raw 126.8 MB text file; the app downloads it from UCI when needed.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
4. Select **Create app**, choose the new repository and `main` branch, set the main file path to `app.py`, then deploy.
5. After deployment, open the generated `*.streamlit.app` URL and verify that the dataset loads, model metrics appear and downloads work.

**Important:** GitHub Pages hosts static websites and cannot run this Python/Streamlit app. Use Streamlit Community Cloud (or another Python-capable host) for the interactive website.

## Method

1. Read the UCI semicolon-separated text file and treat `?` as missing.
2. Combine `Date` and `Time`, parse with day-first dates, and set the timestamp as the index.
3. Convert the seven electrical measurements to numeric values and interpolate missing values over time, followed by forward/backward fill.
4. Resample `Global_active_power` into daily means.
5. Create lag-1, lag-7, lag-14, prior 7-day rolling mean and calendar features.
6. Preserve chronology: the first 80% of the supervised daily rows are training data and the final 20% are test data.
7. Standardise predictors within the Linear Regression pipeline; leave Random Forest predictors unscaled.
8. Compare models with MAE, RMSE and R². Select the lower-RMSE model for the recursive forecast.

The IQR step flags candidate observations; it does not automatically delete high-power points because legitimate high-load events can be meaningful. Normality tests describe the raw power distribution; normally distributed predictors are not a requirement for Random Forest.

## Dataset and citation

Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54. Dataset 235. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

Official dataset page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Known limitations

- Single-household dataset; findings cannot be generalised directly to an entire smart grid.
- The dataset ends in November 2010 and does not provide live meter telemetry.
- Weather, tariff, occupancy and calendar-event covariates are not available in the original project dataset.
- Forecasts are recursive point predictions; no prediction interval or probabilistic uncertainty is calculated.
- The baseline models are suitable for an academic prototype, not operational grid control.

## License

Project code is provided under the MIT License. The UCI dataset retains its own CC BY 4.0 terms; see the citation above.