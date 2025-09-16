import streamlit as st
import pandas as pd
import altair as alt
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

st.set_page_config(page_title="📈 Stock Prediction Dashboard", layout="wide")
st.title("📊 Stock Dashboard: Historical & Predicted Close Prices")

# File path mappings
HISTORICAL_FILES = {
    "META": "META.csv",
    "GOOG": "GOOG.csv",
    "AMZN": "AMZN.csv"
}

PREDICTION_DIRS = {
    "META": "META_Predictions_New",
    "GOOG": "GOOG_Predictions_New",
    "AMZN": "AMZN_Predictions_New"
}


# Stock selection
stock = st.selectbox("Select a Stock", options=["META", "GOOG", "AMZN"])

# Load and display historical chart
if os.path.exists(HISTORICAL_FILES[stock]):
    hist_df = pd.read_csv(HISTORICAL_FILES[stock])
    hist_df['Date'] = pd.to_datetime(hist_df['Date'], errors="coerce")
    hist_df = hist_df.dropna(subset=["Date"])

    st.subheader(f"📘 Historical Data for {stock}")
        # Melt for historical Altair chart
    hist_melted = hist_df.melt(id_vars="Date", value_vars=["Open", "Close", "High", "Low"],
                            var_name="Type", value_name="Price")

    y_min_h = hist_melted["Price"].min() * 0.98
    y_max_h = hist_melted["Price"].max() * 1.02

    hist_chart = alt.Chart(hist_melted).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Price:Q", scale=alt.Scale(domain=[y_min_h, y_max_h]), title="Price"),
        color="Type:N",
        tooltip=["Date:T", "Type:N", "Price:Q"]
    ).properties(
        width=800,
        height=400,
        title=f"{stock} - Historical Stock Prices"
    )

    st.altair_chart(hist_chart, use_container_width=True)

else:
    st.warning("Historical data not found.")

# Load latest prediction file from folder
csv_files = glob.glob(f"{PREDICTION_DIRS[stock]}/*.csv")
if not csv_files:
    st.warning("No prediction file found.")
else:
    latest_file = max(csv_files, key=os.path.getmtime)
    df = pd.read_csv(latest_file)

    # Clean headers and data
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["prediction", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()


    # Melt for Altair
    melted = df[["date", "prediction", "close"]].melt(
        id_vars="date", var_name="Type", value_name="Price"
    )

    st.subheader(f"🔮 Predicted vs Actual Close Price for {stock}")
        # Calculate custom Y-axis domain with padding
    y_min = melted["Price"].min() * 0.98
    y_max = melted["Price"].max() * 1.02

    # Altair Chart with custom scale
    chart = alt.Chart(melted).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Price:Q", scale=alt.Scale(domain=[y_min, y_max]), title="Price"),
        color="Type:N",
        tooltip=["date:T", "Type:N", "Price:Q"]
    ).properties(
        width=800,
        height=400,
        title=f"{stock} - Predicted vs Actual Close"
    )

    st.altair_chart(chart, use_container_width=True)
    # Evaluation section (after chart_df is ready)
    y_true = df['close']
    y_pred = df['prediction']

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.subheader("📈 Model Evaluation Metrics (Last 3 Months)")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("R²", f"{r2:.4f}")


    # st.subheader("📋 Last 10 Predictions")
    # st.dataframe(df[["date", "prediction", "close"]].tail(10))

    # Optional: Download CSV
    csv = df[["date", "prediction", "close"]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions", csv, f"{stock}_predictions.csv", "text/csv")
    
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("📊 Additional Visualizations")

tabs = st.tabs(["META", "GOOG", "AMZN"])

stocks = {
    "META": {
        "csv": "META_last_3_months.csv",
        "pred": "META_Predictions_New/part-00000-1d410438-d439-4590-811c-78b36c11e09c-c000.csv"
    },
    "GOOG": {
        "csv": "GOOG_last_3_months.csv",
        "pred": "GOOG_Predictions_New/part-00000-f8b33759-c016-4b76-9905-c3adb2d51b45-c000.csv"
    },
    "AMZN": {
        "csv": "AMZN_last_3_months.csv",
        "pred": "AMZN_Predictions_New/part-00000-16d33f1a-5468-4a7a-9de1-f2f0c93f3d59-c000.csv"
    }
}

for tab, (stock_name, paths) in zip(tabs, stocks.items()):
    with tab:
        st.markdown(f"### 📈 {stock_name} Candlestick with Volume")
        try:
            df = pd.read_csv(paths["csv"])
            df['date'] = pd.to_datetime(df['date'])

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price"
            ))

            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.3,
                yaxis='y2'
            ))

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                yaxis2=dict(overlaying='y', side='right', title='Volume', showgrid=False),
                xaxis_rangeslider_visible=False,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load candlestick chart for {stock_name}: {e}")

        st.markdown(f"### 📍 {stock_name} Predicted vs Actual Close")
        try:
            pred_df = pd.read_csv(paths["pred"])
            pred_df["prediction"] = pd.to_numeric(pred_df["prediction"], errors="coerce")
            pred_df["Close"] = pd.to_numeric(pred_df["Close"], errors="coerce")
            pred_df = pred_df.dropna(subset=["prediction", "Close"])

            scatter_fig, ax = plt.subplots(figsize=(5, 5))
            sns.scatterplot(x=pred_df["Close"], y=pred_df["prediction"], alpha=0.7, ax=ax)
            ax.plot([pred_df["Close"].min(), pred_df["Close"].max()],
                    [pred_df["Close"].min(), pred_df["Close"].max()], 'r--')
            ax.set_xlabel("Actual Close")
            ax.set_ylabel("Predicted Close")
            ax.set_title(f"{stock_name}: Predicted vs Actual")
            ax.grid(True)
            plt.tight_layout()

            left, middle, right = st.columns([1, 2, 1])  # middle column is 1/2 width
            with middle:
                st.pyplot(scatter_fig)

        except Exception as e:
            st.warning(f"Could not load prediction scatter plot for {stock_name}: {e}")
