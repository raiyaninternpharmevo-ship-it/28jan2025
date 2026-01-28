import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet("sales.parquet")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["ValueNp"] = pd.to_numeric(df["ValueNp"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "ValueNp"])
    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.title("🔎 Filters")

client_type = st.sidebar.multiselect(
    "Client Type",
    options=df["ClientType"].unique(),
    default=df["ClientType"].unique()
)

date_range = st.sidebar.date_input(
    "Invoice Date Range",
    [df["InvoiceDate"].min(), df["InvoiceDate"].max()]
)

df = df[
    (df["ClientType"].isin(client_type)) &
    (df["InvoiceDate"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 Sales Intelligence Dashboard")
st.caption("Industry-grade analytics for management decision making")

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
total_sales = df["ValueNp"].sum()
total_invoices = df["InvoiceNo"].nunique()
avg_invoice = total_sales / total_invoices if total_invoices else 0

monthly_sales = df.groupby(df["InvoiceDate"].dt.to_period("M"))["ValueNp"].sum()
growth = (
    (monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2] * 100
    if len(monthly_sales) > 1 else 0
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"{total_sales:,.0f}")
k2.metric("Total Invoices", f"{total_invoices:,}")
k3.metric("Avg Invoice Value", f"{avg_invoice:,.0f}")
k4.metric("Monthly Growth", f"{growth:.1f}%")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈 Overview", "📦 Products", "🔮 Forecast"])

# --------------------------------------------------
# TAB 1 — OVERVIEW
# --------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sales by Distributor")
        fig = px.bar(
            df.groupby("DistributorName")["ValueNp"].sum().reset_index(),
            x="DistributorName",
            y="ValueNp",
            color="ValueNp",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sales by Client Type")
        fig = px.pie(
            df.groupby("ClientType")["ValueNp"].sum().reset_index(),
            names="ClientType",
            values="ValueNp",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Sales Trend")
    monthly = monthly_sales.reset_index()
    monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="ValueNp",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 2 — PRODUCTS
# --------------------------------------------------
with tab2:
    st.subheader("Top 10 Products by Sales")

    top_products = (
        df.groupby("ProductName")["ValueNp"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="ProductName",
        y="ValueNp",
        color="ValueNp",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TAB 3 — FORECASTING (EMA)
# --------------------------------------------------
with tab3:
    st.subheader("Sales Forecast (Next 6 Months)")
    st.caption("Method: Exponential Moving Average (Industry-safe baseline model)")

    forecast_months = st.slider("Forecast Horizon (Months)", 3, 12, 6)

    ts = monthly_sales.copy()
    ts.index = ts.index.to_timestamp()

    # EMA Forecast
    ema = ts.ewm(span=3).mean()
    last_value = ema.iloc[-1]

    future_dates = pd.date_range(
        start=ts.index[-1] + pd.offsets.MonthBegin(),
        periods=forecast_months,
        freq="MS"
    )

    forecast = pd.Series(
        [last_value] * forecast_months,
        index=future_dates
    )

    forecast_df = pd.concat([
        ts.rename("Actual"),
        forecast.rename("Forecast")
    ], axis=1)

    fig = px.line(
        forecast_df,
        labels={"value": "Sales Value", "index": "Month"},
    )
    fig.update_layout(legend_title_text="")

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "📌 Forecast represents expected stable trend based on recent performance. "
        "Advanced ML models can be added for seasonality and promotions."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© Sales Analytics | Built with Streamlit & Plotly")
