"""Reusable data, diagnostics and forecasting functions for the Smart Grid project."""
from __future__ import annotations
from io import BytesIO
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
MEASURE_COLUMNS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
                   "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
FEATURES = ["Lag_1", "Lag_7", "Lag_14", "Rolling_7", "Day", "Month", "DayOfWeek"]


def parse_household_file(payload: bytes) -> pd.DataFrame:
    """Read either the original semicolon TXT file or the UCI ZIP archive."""
    source = BytesIO(payload)
    if zipfile.is_zipfile(source):
        source.seek(0)
        with zipfile.ZipFile(source) as archive:
            files = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if not files:
                raise ValueError("The ZIP archive does not contain the household TXT dataset.")
            source = BytesIO(archive.read(files[0]))
    source.seek(0)
    frame = pd.read_csv(source, sep=";", na_values="?", low_memory=False)
    required = {"Date", "Time", *MEASURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")
    frame["Datetime"] = pd.to_datetime(frame["Date"].astype(str) + " " + frame["Time"].astype(str),
                                      dayfirst=True, errors="coerce")
    frame = frame.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")].drop(columns=["Date", "Time"])
    for col in MEASURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[MEASURE_COLUMNS] = frame[MEASURE_COLUMNS].interpolate(method="time").ffill().bfill()
    if frame[MEASURE_COLUMNS].isna().any().any():
        raise ValueError("Missing measurement values remain after interpolation.")
    return frame


def daily_target(cleaned: pd.DataFrame) -> pd.Series:
    series = cleaned["Global_active_power"].resample("D").mean().dropna()
    series.name = "Power"
    return series


def make_supervised(daily: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Power": daily})
    frame["Lag_1"] = frame["Power"].shift(1)
    frame["Lag_7"] = frame["Power"].shift(7)
    frame["Lag_14"] = frame["Power"].shift(14)
    frame["Rolling_7"] = frame["Power"].shift(1).rolling(7).mean()
    frame["Day"], frame["Month"], frame["DayOfWeek"] = frame.index.day, frame.index.month, frame.index.dayofweek
    return frame.dropna()


def train_and_evaluate(daily: pd.Series) -> dict[str, Any]:
    """Evaluate on chronological final 20%, then refit the winner on all available rows."""
    supervised = make_supervised(daily)
    if len(supervised) < 40:
        raise ValueError("At least 40 usable daily observations are required.")
    split = int(len(supervised) * 0.8)
    X, y = supervised[FEATURES], supervised["Power"]
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    factories = {
        "Linear Regression": lambda: Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    metrics, predictions = [], {}
    for name, factory in factories.items():
        model = factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics.append({"Model": name, "MAE": float(mean_absolute_error(y_test, pred)),
                        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                        "R²": float(r2_score(y_test, pred))})
        predictions[name + " Predicted"] = pd.Series(pred, index=y_test.index)
    metric_frame = pd.DataFrame(metrics).sort_values("RMSE").reset_index(drop=True)
    winner = str(metric_frame.iloc[0]["Model"])
    final_model = factories[winner]()
    final_model.fit(X, y)
    return {"metrics": metric_frame, "predictions": pd.DataFrame({"Actual": y_test, **predictions}),
            "best_model_name": winner, "best_model": final_model,
            "supervised": supervised, "split_date": y_test.index.min()}


def forecast_daily(daily: pd.Series, model: Any, horizon: int) -> pd.DataFrame:
    if not 1 <= int(horizon) <= 90:
        raise ValueError("Forecast horizon must be between 1 and 90 days.")
    history, last_date, rows = list(daily.astype(float).values), pd.Timestamp(daily.index[-1]), []
    for step in range(1, int(horizon) + 1):
        if len(history) < 14:
            raise ValueError("At least 14 days are required for forecasting.")
        date = last_date + pd.Timedelta(days=step)
        features = pd.DataFrame([{"Lag_1": history[-1], "Lag_7": history[-7], "Lag_14": history[-14],
            "Rolling_7": float(np.mean(history[-7:])), "Day": date.day, "Month": date.month,
            "DayOfWeek": date.dayofweek}], index=[date])[FEATURES]
        value = max(0.0, float(model.predict(features)[0]))
        history.append(value)
        rows.append({"Date": date, "Forecast_kW": value})
    return pd.DataFrame(rows).set_index("Date")


def diagnostics(values: pd.Series, random_state: int = 42) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("No numeric values are available for diagnostics.")
    sample = clean.sample(n=min(5000, len(clean)), random_state=random_state)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    dag_stat, dag_p = stats.normaltest(sample)
    q1, q3 = clean.quantile([.25, .75])
    iqr = float(q3 - q1)
    lower, upper = float(q1 - 1.5*iqr), float(q3 + 1.5*iqr)
    mask = (clean < lower) | (clean > upper)
    return {"sample": sample, "shapiro_stat": float(shapiro_stat), "shapiro_p": float(shapiro_p),
        "dagostino_stat": float(dag_stat), "dagostino_p": float(dag_p), "skewness": float(clean.skew()),
        "kurtosis_excess": float(clean.kurtosis()), "q1": float(q1), "q3": float(q3), "iqr": iqr,
        "lower_bound": lower, "upper_bound": upper, "outlier_count": int(mask.sum()),
        "outlier_pct": float(mask.mean()*100), "total_count": int(len(clean))}
