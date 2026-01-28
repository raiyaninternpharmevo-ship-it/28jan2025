import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_parquet("sales.parquet")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["ValueNp"] = pd.to_numeric(df["ValueNp"], errors="coerce")
    return df

df = load_data()

st.title("📊 Sales Intelligence Dashboard")
st.markdown("### Business-ready insights for management")

# Sidebar filters
st.sidebar.header("Filters")
client_type = st.sidebar.multiselect(
    "Select Client Type",
    df["ClientType"].unique(),
    default=df["ClientType"].unique()
)

df = df[df["ClientType"].isin(client_type)]

# KPI Cards
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{df['ValueNp'].sum():,.0f}")
col2.metric("Total Invoices", df["InvoiceNo"].nunique())
col3.metric("Total Products", df["ProductName"].nunique())

# Charts
st.subheader("Sales by Distributor")
fig1 = px.bar(
    df.groupby("DistributorName")["ValueNp"].sum().reset_index(),
    x="DistributorName",
    y="ValueNp"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Sales by Client Type")
fig2 = px.pie(
    df.groupby("ClientType")["ValueNp"].sum().reset_index(),
    names="ClientType",
    values="ValueNp"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Products")
fig3 = px.bar(
    df.groupby("ProductName")["ValueNp"].sum().sort_values(ascending=False).head(10).reset_index(),
    x="ProductName",
    y="ValueNp"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Monthly Sales Trend")
monthly = df.groupby(df["InvoiceDate"].dt.to_period("M"))["ValueNp"].sum().reset_index()
monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)

fig4 = px.line(monthly, x="InvoiceDate", y="ValueNp", markers=True)
st.plotly_chart(fig4, use_container_width=True)
