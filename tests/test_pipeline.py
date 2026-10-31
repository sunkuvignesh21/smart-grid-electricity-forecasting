import numpy as np
import pandas as pd
from src.pipeline import MEASURE_COLUMNS, make_supervised, train_and_evaluate, forecast_daily, parse_household_file


def synthetic_daily(days=240):
    idx = pd.date_range("2020-01-01", periods=days, freq="D")
    rng = np.random.default_rng(7)
    values = 1.2 + .15*np.sin(np.arange(days)*2*np.pi/7) + rng.normal(0,.025,days)
    return pd.Series(values, index=idx, name="Power")


def test_features_models_and_forecast():
    daily = synthetic_daily()
    frame = make_supervised(daily)
    assert not frame.isna().any().any()
    result = train_and_evaluate(daily)
    assert set(result["metrics"]["Model"]) == {"Linear Regression", "Random Forest"}
    future = forecast_daily(daily, result["best_model"], 7)
    assert len(future) == 7 and (future["Forecast_kW"] >= 0).all()


def test_parse_interpolates_missing_measurement():
    header = "Date;Time;" + ";".join(MEASURE_COLUMNS) + "\n"
    rows = ["16/12/2006;17:24:00;1.0;0.2;240;5;0;0;17\n",
            "16/12/2006;17:25:00;?;0.2;240;5;0;0;17\n",
            "16/12/2006;17:26:00;3.0;0.2;240;5;0;0;17\n"]
    df = parse_household_file((header + "".join(rows)).encode())
    assert np.isclose(df["Global_active_power"].iloc[1], 2.0)
