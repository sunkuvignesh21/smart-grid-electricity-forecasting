from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.pipeline import (
    DATA_URL,
    FEATURES,
    MEASURE_COLUMNS,
    daily_target,
    diagnostics,
    forecast_daily,
    parse_household_file,
    train_and_evaluate,
)

st.set_page_config(
    page_title="Smart Grid | Electricity Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px;}
      h1, h2, h3 {letter-spacing: -0.025em;}
      .hero {background: linear-gradient(115deg,#102A43 0%,#145C63 58%,#0E8A74 100%);
             border-radius: 18px; padding: 1.55rem 1.8rem; color: white; margin-bottom: 1.15rem;}
      .hero h1 {color:white; margin:0 0 .35rem 0; font-size:2.15rem;}
      .hero p {color:#DCEEEB; margin:0; font-size:1rem;}
      div[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E5EAF0;
        padding:14px 16px; border-radius:13px; box-shadow:0 2px 8px rgba(20,40,60,.035);}
      div[data-testid="stMetricLabel"] p {color:#52616B;}
      .section-note {color:#596A75; font-size:.91rem;}
      .pill {display:inline-block; background:#E5F5F1; color:#086B5D; padding:4px 10px;
             border-radius:999px; font-weight:600; font-size:.78rem;}
      [data-testid="stSidebar"] {background:#F0F4F7;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=2)
def get_uci_archive() -> bytes:
    response = requests.get(DATA_URL, timeout=180)
    response.raise_for_status()
    return response.content


@st.cache_data(show_spinner=False, max_entries=2)
def cached_parse(payload: bytes) -> pd.DataFrame:
    return parse_household_file(payload)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_model_results(daily: pd.Series):
    return train_and_evaluate(daily)


def fmt(value: float, decimals: int = 3) -> str:
    return f"{value:,.{decimals}f}"


def show_figure(fig, key: str | None = None):
    # Hide the Plotly modebar to keep charts uncluttered on narrow mobile screens.
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={"displayModeBar": False, "responsive": True},
    )


st.markdown(
    """
    <div class="hero">
      <span class="pill">MACHINE LEARNING · TIME SERIES</span>
      <h1>Smart Grid Electricity Analytics</h1>
      <p>Explore household electricity patterns, inspect data quality, compare forecasting models,
      and generate short-term daily power forecasts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data setup")
    source_choice = st.radio(
        "Choose dataset source",
        ["Download official UCI dataset", "Upload household_power_consumption.txt"],
        index=0,
    )
    uploaded = None
    if source_choice.startswith("Upload"):
        uploaded = st.file_uploader(
            "Select the original semicolon-separated TXT file",
            type=["txt", "csv", "zip"],
            help="The original UCI .txt file or its ZIP archive is supported.",
        )
    st.divider()
    st.caption("Dataset: Individual Household Electric Power Consumption")
    st.caption("Source: UCI Machine Learning Repository · Dataset 235")
    st.link_button("Open UCI dataset page", "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption", use_container_width=True)
    st.divider()
    st.caption("Forecast target: daily mean Global_active_power (kW)")

try:
    if source_choice.startswith("Download"):
        with st.spinner("Downloading the official UCI archive (about 20 MB)…"):
            payload = get_uci_archive()
    else:
        if uploaded is None:
            st.info("Upload the original UCI TXT or ZIP file in the sidebar, or switch to the official UCI download.")
            st.stop()
        payload = uploaded.getvalue()
    with st.spinner("Parsing, cleaning and interpolating the minute-level readings…"):
        cleaned = cached_parse(payload)
        daily = daily_target(cleaned)
except requests.RequestException as exc:
    st.error(f"Could not download the UCI dataset. Check your connection or upload the TXT file instead. Details: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

if len(daily) < 40:
    st.error("The dataset contains too few daily records to train and evaluate the forecasting models.")
    st.stop()

# Calculate the modelling results once for the currently selected data.
try:
    with st.spinner("Training Linear Regression and Random Forest on a chronological 80/20 split…"):
        results = cached_model_results(daily)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

metrics = results["metrics"]
predictions = results["predictions"]
best_model_name = results["best_model_name"]
best_model = results["best_model"]
diagnostic_results = diagnostics(cleaned["Global_active_power"])

# Sidebar summary
with st.sidebar:
    st.success("Dataset loaded and models evaluated")
    st.metric("Minute readings", f"{len(cleaned):,}")
    st.metric("Daily records", f"{len(daily):,}")
    st.metric("Best holdout model", best_model_name)

# Overview KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Date coverage", f"{daily.index.min():%b '%y} – {daily.index.max():%b '%y}")
c2.metric("Average daily power", f"{daily.mean():.3f} kW")
c3.metric("Peak minute power", f"{cleaned['Global_active_power'].max():.2f} kW")
best_rmse = float(metrics.loc[metrics["Model"] == best_model_name, "RMSE"].iloc[0])
c4.metric("Best test RMSE", f"{best_rmse:.4f} kW")

st.caption(
    "Scope note: this project analyses one household's historical measurements. "
    "It is an educational forecasting prototype, not a live utility-grid monitoring system."
)

tabs = st.tabs(["Overview", "Consumption patterns", "Data diagnostics", "Forecasting", "Dataset & method"])

with tabs[0]:
    left, right = st.columns([1.65, 1])
    with left:
        st.subheader("Daily mean active power")
        recent_days = st.selectbox("History window", ["All available dates", "Last 365 days", "Last 180 days", "Last 90 days"], index=0)
        view = daily if recent_days == "All available dates" else daily.tail(int(recent_days.split()[1]))
        daily_view = pd.DataFrame({
            "Date": pd.to_datetime(view.index),
            "Daily mean active power": view.to_numpy(),
        })
        fig = px.line(daily_view, x="Date", y="Daily mean active power")
        fig.update_traces(line=dict(width=1.6, color="#0E8A74"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Power (kW)", margin=dict(l=10,r=10,t=20,b=10), hovermode="x unified")
        show_figure(fig, "overview_daily")
    with right:
        st.subheader("Model comparison")
        bar = px.bar(metrics.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error"),
                     x="Metric", y="Error", color="Model", barmode="group", text_auto=".3f")
        bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), legend_title="Model", yaxis_title="Error (kW)")
        show_figure(bar, "overview_model_comparison")
        st.markdown("**Best holdout model:** " + best_model_name)
        st.caption("Selection is based on the lower RMSE on the final chronological test period.")
    st.subheader("Actual vs predicted on the test period")
    pred_chart = go.Figure()
    pred_chart.add_trace(go.Scatter(x=predictions.index, y=predictions["Actual"], mode="lines", name="Actual", line=dict(color="#243B53", width=2)))
    for name, color in [("Linear Regression Predicted", "#0E8A74"), ("Random Forest Predicted", "#E39A3B")]:
        pred_chart.add_trace(go.Scatter(x=predictions.index, y=predictions[name], mode="lines", name=name.replace(" Predicted", ""), line=dict(width=1.5, color=color)))
    pred_chart.update_layout(xaxis_title="Date", yaxis_title="Daily mean active power (kW)", hovermode="x unified", margin=dict(l=10,r=10,t=20,b=10))
    show_figure(pred_chart, "overview_actual_predicted")

with tabs[1]:
    st.subheader("When is electricity use highest?")
    a, b = st.columns(2)
    with a:
        hourly = cleaned.groupby(cleaned.index.hour)["Global_active_power"].mean().rename("Mean power")
        hourly_frame = hourly.rename_axis("Hour").reset_index()
        fig_hour = px.bar(hourly_frame, x="Hour", y="Mean power", color="Mean power", color_continuous_scale="Teal")
        fig_hour.update_layout(xaxis=dict(dtick=1), xaxis_title="Hour of day", yaxis_title="Mean active power (kW)", coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=10))
        show_figure(fig_hour, "hourly_pattern")
    with b:
        weekday = cleaned.groupby(cleaned.index.dayofweek)["Global_active_power"].mean()
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        week_frame = pd.DataFrame({"Day": day_names, "Mean power": [weekday.get(i, np.nan) for i in range(7)]})
        fig_week = px.bar(week_frame, x="Day", y="Mean power", color="Mean power", color_continuous_scale="Teal")
        fig_week.update_layout(xaxis_title="Day of week", yaxis_title="Mean active power (kW)", coloraxis_showscale=False, margin=dict(l=10,r=10,t=20,b=10))
        show_figure(fig_week, "weekday_pattern")
    c, d = st.columns(2)
    with c:
        monthly = cleaned["Global_active_power"].resample("MS").mean().dropna()
        monthly_view = pd.DataFrame({"Date": monthly.index, "Mean power": monthly.to_numpy()})
        fig_month = px.line(monthly_view, x="Date", y="Mean power", markers=True)
        fig_month.update_traces(line_color="#0E8A74")
        fig_month.update_layout(xaxis_title="Month", yaxis_title="Mean active power (kW)", margin=dict(l=10,r=10,t=20,b=10))
        st.subheader("Monthly pattern")
        show_figure(fig_month, "monthly_pattern")
    with d:
        weekly = daily.resample("W").mean().dropna()
        weekly_view = pd.DataFrame({"Date": weekly.index, "Mean power": weekly.to_numpy()})
        fig_weekly = px.line(weekly_view, x="Date", y="Mean power")
        fig_weekly.update_traces(line_color="#E39A3B")
        fig_weekly.update_layout(xaxis_title="Week ending", yaxis_title="Mean active power (kW)", margin=dict(l=10,r=10,t=20,b=10))
        st.subheader("Weekly pattern")
        show_figure(fig_weekly, "weekly_pattern")
    st.subheader("Correlation between electrical measurements")
    corr = cleaned[MEASURE_COLUMNS].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(margin=dict(l=10,r=10,t=20,b=10), coloraxis_colorbar_title="Pearson r")
    show_figure(fig_corr, "correlation_heatmap")

with tabs[2]:
    st.subheader("Normality assessment — Global active power")
    st.markdown("Tests use a reproducible sample of up to 5,000 minute readings; skewness, kurtosis and IQR bounds use all cleaned readings.")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Shapiro–Wilk p-value", f"{diagnostic_results['shapiro_p']:.3g}")
    n2.metric("D’Agostino p-value", f"{diagnostic_results['dagostino_p']:.3g}")
    n3.metric("Skewness", f"{diagnostic_results['skewness']:.3f}")
    n4.metric("Excess kurtosis", f"{diagnostic_results['kurtosis_excess']:.3f}")
    st.caption("A p-value below 0.05 provides evidence against normality. A large sample can make these tests sensitive to small deviations.")

    hist_col, qq_col = st.columns(2)
    with hist_col:
        st.markdown("**Distribution histogram**")
        hist_counts, hist_edges = np.histogram(cleaned["Global_active_power"].to_numpy(), bins=55)
        hist_centres = (hist_edges[:-1] + hist_edges[1:]) / 2
        fig_hist = go.Figure(go.Bar(x=hist_centres, y=hist_counts, marker_color="#7BB3C7",
                                    width=np.diff(hist_edges)))
        fig_hist.update_layout(xaxis_title="Global active power (kW)", yaxis_title="Count",
                               margin=dict(l=10,r=10,t=20,b=10))
        show_figure(fig_hist, "normality_hist")
    with qq_col:
        st.markdown("**Normal Q–Q plot**")
        sample = diagnostic_results["sample"].values
        theoretical, ordered = stats.probplot(sample, dist="norm", fit=False)
        qq = go.Figure()
        qq.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers", name="Sample quantiles", marker=dict(color="#2457D6", size=4)))
        fit = stats.linregress(theoretical, ordered)
        xline = np.array([min(theoretical), max(theoretical)])
        qq.add_trace(go.Scatter(x=xline, y=fit.intercept + fit.slope*xline, mode="lines", name="Reference line", line=dict(color="red", dash="dash")))
        qq.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Ordered values", margin=dict(l=10,r=10,t=20,b=10))
        show_figure(qq, "normality_qq")

    st.subheader("IQR outlier assessment")
    o1, o2, o3 = st.columns(3)
    o1.metric("IQR outlier count", f"{diagnostic_results['outlier_count']:,}")
    o2.metric("Outlier proportion", f"{diagnostic_results['outlier_pct']:.2f}%")
    o3.metric("Upper IQR bound", f"{diagnostic_results['upper_bound']:.3f} kW")
    st.write(f"Lower bound: **{diagnostic_results['lower_bound']:.3f} kW** · Q1: **{diagnostic_results['q1']:.3f} kW** · Q3: **{diagnostic_results['q3']:.3f} kW**")
    box = go.Figure(go.Box(x=diagnostic_results["sample"], boxpoints="outliers",
                           name="Diagnostic sample (n ≤ 5,000)", marker_color="#DDA33F"))
    box.add_vline(x=diagnostic_results["lower_bound"], line_dash="dash", line_color="#D7263D", annotation_text="Lower IQR bound")
    box.add_vline(x=diagnostic_results["upper_bound"], line_dash="dash", line_color="#D7263D", annotation_text="Upper IQR bound")
    box.update_layout(xaxis_title="Global active power (kW)", showlegend=False, margin=dict(l=10,r=10,t=25,b=10))
    show_figure(box, "iqr_boxplot")
    st.warning("IQR flags are statistical candidates, not proof of faulty readings. Consumption peaks are retained because genuine high-load periods may be important for forecasting.")

    st.subheader("Scaling comparison")
    scale_sample = cleaned["Global_active_power"].sample(n=min(12000, len(cleaned)), random_state=42).to_numpy().reshape(-1, 1)
    standard = StandardScaler().fit_transform(scale_sample).ravel()
    minmax = MinMaxScaler().fit_transform(scale_sample).ravel()
    scaled_df = pd.DataFrame({
        "Original (kW)": scale_sample.ravel(),
        "Standardised (z-score)": standard,
        "Min–max normalised": minmax,
    }).melt(var_name="Transformation", value_name="Value")
    scale_fig = px.histogram(scaled_df, x="Value", facet_col="Transformation", facet_col_wrap=1, nbins=45, color="Transformation")
    scale_fig.update_layout(showlegend=False, height=650, margin=dict(l=10,r=10,t=25,b=10))
    scale_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    show_figure(scale_fig, "scaling_comparison")
    st.info("StandardScaler is applied to Linear Regression inside a training-only pipeline. Random Forest is not scaled because tree split rules do not depend on feature magnitude.")

with tabs[3]:
    st.subheader("Short-term daily forecast")
    st.markdown(f"The forecast uses **{best_model_name}**, selected by the lower RMSE on the held-out final 20% of the time series.")
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)
    future = forecast_daily(daily, best_model, horizon)
    hist_plot = daily.tail(120)
    fc_fig = go.Figure()
    fc_fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot.values, mode="lines", name="Historical", line=dict(color="#243B53", width=2)))
    fc_fig.add_trace(go.Scatter(x=future.index, y=future["Forecast_kW"], mode="lines+markers", name="Forecast", line=dict(color="#0E8A74", width=2, dash="dash")))
    fc_fig.add_vline(x=daily.index[-1], line_dash="dot", line_color="#8293A1")
    fc_fig.update_layout(xaxis_title="Date", yaxis_title="Daily mean active power (kW)", hovermode="x unified", margin=dict(l=10,r=10,t=20,b=10))
    show_figure(fc_fig, "future_forecast")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Forecast average", f"{future['Forecast_kW'].mean():.3f} kW")
    fc2.metric("Forecast minimum", f"{future['Forecast_kW'].min():.3f} kW")
    fc3.metric("Forecast maximum", f"{future['Forecast_kW'].max():.3f} kW")
    st.dataframe(future.reset_index().rename(columns={"Forecast_kW": "Forecast (kW)"}), use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV", future.to_csv().encode("utf-8"),
        file_name="smart_grid_future_forecast.csv", mime="text/csv",
    )
    st.caption("Forecasts are recursive: each predicted day is used to construct the lag features for the next day. Uncertainty intervals are not estimated in this baseline implementation.")

    st.subheader("Holdout model scores")
    st.dataframe(metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.download_button(
        "Download model evaluation CSV", metrics.to_csv(index=False).encode("utf-8"),
        file_name="smart_grid_model_metrics.csv", mime="text/csv",
    )
    export = predictions.copy()
    st.download_button(
        "Download actual vs predicted CSV", export.to_csv(index_label="Date").encode("utf-8"),
        file_name="daily_electricity_forecasting.csv", mime="text/csv",
    )

with tabs[4]:
    st.subheader("Dataset overview")
    st.write("The UCI dataset contains one-minute electricity measurements from a single household in Sceaux, France, from December 2006 to November 2010. It contains 2,075,259 records and nine original columns.")
    schema_rows = [
        ("Date", "Date", "Date in dd/mm/yyyy format"),
        ("Time", "Categorical / time", "Time in hh:mm:ss format"),
        ("Global_active_power", "Continuous", "Minute-averaged active power (kW)"),
        ("Global_reactive_power", "Continuous", "Minute-averaged reactive power (kW)"),
        ("Voltage", "Continuous", "Minute-averaged voltage (V)"),
        ("Global_intensity", "Continuous", "Minute-averaged current (A)"),
        ("Sub_metering_1", "Continuous", "Kitchen sub-metering (Wh)"),
        ("Sub_metering_2", "Continuous", "Laundry-room sub-metering (Wh)"),
        ("Sub_metering_3", "Continuous", "Water-heater / air-conditioning sub-metering (Wh)"),
    ]
    st.dataframe(pd.DataFrame(schema_rows, columns=["Feature", "Type", "Description"]), use_container_width=True, hide_index=True)
    st.subheader("Cleaning and feature engineering")
    st.markdown(
        """
        - Parse `Date` + `Time` into a datetime index and convert measurement columns to numeric values.
        - Treat `?` as missing and fill measurement gaps with time-based interpolation, followed by forward/backward fill.
        - Aggregate `Global_active_power` into a daily mean target in kW.
        - Construct `Lag_1`, `Lag_7`, `Lag_14`, prior 7-day rolling mean, and calendar features (day, month, weekday).
        - Preserve temporal order: first 80% for training, final 20% for testing.
        """
    )
    st.subheader("Model choices and evaluation")
    st.markdown("Linear Regression is a transparent baseline and receives standardised predictors. Random Forest captures non-linear relationships and does not require scaling. Both models are evaluated with MAE, RMSE and R² on the same chronological holdout.")
    st.dataframe(metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.subheader("Data preview")
    st.dataframe(cleaned.head(10), use_container_width=True)
    st.download_button(
        "Download cleaned daily series", daily.rename("Power_kW").to_csv().encode("utf-8"),
        file_name="daily_electricity_series.csv", mime="text/csv",
    )
    st.subheader("Reproducibility and limitations")
    st.markdown(
        "The UCI data is historical and describes one household only. There is no live smart-meter feed, weather input, multi-household generalisation, or uncertainty interval in this prototype. Results should not be interpreted as forecasts for a regional electricity grid. The minute-level dataset is downloaded at runtime and is not stored in this Git repository."
    )
    st.markdown("**Dataset citation:** Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. [https://doi.org/10.24432/C58K54](https://doi.org/10.24432/C58K54). Licensed CC BY 4.0.")

st.divider()
st.caption("Smart Grid Electricity Consumption Analysis and Forecasting · Academic prototype · Built with Python, pandas, scikit-learn and Streamlit")