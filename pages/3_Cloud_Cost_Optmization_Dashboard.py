from datetime import datetime, timedelta
import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Cloud Cost Optimization Dashboard",
    page_icon="💸",
    layout="wide",
)


# ---------------------------------------------------
# Safe display helpers
# ---------------------------------------------------
def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value)


def make_display_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(safe_str)
    return out


def render_html_table(df: pd.DataFrame, height: int = 380):
    if df.empty:
        st.info("No data available.")
        return

    safe_df = make_display_safe(df)
    html = safe_df.to_html(index=False, escape=False)

    styled_html = f"""
    <div style="
        max-height:{height}px;
        overflow:auto;
        border:1px solid #ddd;
        border-radius:8px;
        padding:8px;
        background-color:white;
    ">
        {html}
    </div>
    """

    st.markdown(
        """
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #e6e6e6;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background-color: #f7f7f7;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(styled_html, unsafe_allow_html=True)


def render_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    if series.empty:
        st.info("No chart data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(series.index.astype(str), series.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def render_line_chart(df: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    if df.empty:
        st.info("No chart data available.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in df.columns:
        ax.plot(df.index.astype(str), df[col], marker="o", label=str(col))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------
# Sample data generation
# ---------------------------------------------------
@st.cache_data
def generate_cloud_cost_data(days: int = 30) -> pd.DataFrame:
    services = [
        {"service": "EC2", "base_cost": 42, "cpu_avg": 18, "memory_avg": 34, "storage_gb": 120, "env": "Production"},
        {"service": "RDS", "base_cost": 56, "cpu_avg": 22, "memory_avg": 48, "storage_gb": 300, "env": "Production"},
        {"service": "S3", "base_cost": 18, "cpu_avg": 0, "memory_avg": 0, "storage_gb": 2500, "env": "Shared"},
        {"service": "BigQuery", "base_cost": 64, "cpu_avg": 0, "memory_avg": 0, "storage_gb": 900, "env": "Analytics"},
        {"service": "Snowflake", "base_cost": 51, "cpu_avg": 0, "memory_avg": 0, "storage_gb": 750, "env": "Analytics"},
        {"service": "Databricks", "base_cost": 47, "cpu_avg": 26, "memory_avg": 40, "storage_gb": 500, "env": "Analytics"},
        {"service": "Redis", "base_cost": 14, "cpu_avg": 12, "memory_avg": 22, "storage_gb": 30, "env": "Production"},
        {"service": "CloudWatch", "base_cost": 9, "cpu_avg": 0, "memory_avg": 0, "storage_gb": 150, "env": "Shared"},
    ]

    start_date = datetime.today().date() - timedelta(days=days - 1)
    rows = []

    for i in range(days):
        current_date = start_date + timedelta(days=i)

        for svc in services:
            daily_cost = round(max(1, random.gauss(svc["base_cost"], svc["base_cost"] * 0.18)), 2)

            cpu = round(max(0, min(100, random.gauss(svc["cpu_avg"], 8))), 2) if svc["cpu_avg"] else None
            memory = round(max(0, min(100, random.gauss(svc["memory_avg"], 10))), 2) if svc["memory_avg"] else None
            query_scan_gb = round(max(0, random.gauss(220, 80)), 2) if svc["service"] in ["BigQuery", "Snowflake"] else None
            idle_hours = round(max(0, random.gauss(6, 2.5)), 2) if svc["service"] in ["EC2", "Databricks", "Redis"] else None

            rows.append(
                {
                    "usage_date": str(current_date),
                    "service": svc["service"],
                    "environment": svc["env"],
                    "daily_cost_usd": daily_cost,
                    "cpu_utilization_pct": cpu,
                    "memory_utilization_pct": memory,
                    "storage_gb": svc["storage_gb"],
                    "query_scan_gb": query_scan_gb,
                    "idle_hours": idle_hours,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------
# Optimization logic
# ---------------------------------------------------
def classify_cost_issues(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    recommendations = []
    issue_type = []
    estimated_saving = []

    for _, row in out.iterrows():
        svc = row["service"]
        cpu = row["cpu_utilization_pct"]
        memory = row["memory_utilization_pct"]
        scan_gb = row["query_scan_gb"]
        idle_hours = row["idle_hours"]
        cost = row["daily_cost_usd"]

        rec = "No major issue detected"
        issue = "Healthy"
        saving = 0.0

        if svc in ["EC2", "RDS", "Databricks", "Redis"]:
            if pd.notna(cpu) and pd.notna(memory):
                if cpu < 20 and memory < 40:
                    issue = "Underutilized Compute"
                    rec = "Consider rightsizing instance/cluster to a smaller size."
                    saving = round(cost * 0.25, 2)
                elif cpu > 85:
                    issue = "High Compute Load"
                    rec = "Evaluate autoscaling or workload balancing to avoid expensive burst behavior."
                    saving = round(cost * 0.05, 2)

        if svc in ["BigQuery", "Snowflake"] and pd.notna(scan_gb):
            if scan_gb > 250:
                issue = "High Query Scan"
                rec = "Review partitioning, clustering, predicate pushdown, and SELECT column pruning."
                saving = round(cost * 0.30, 2)

        if svc in ["EC2", "Databricks", "Redis"] and pd.notna(idle_hours):
            if idle_hours > 6:
                issue = "High Idle Time"
                rec = "Schedule shutdown outside business hours or pause non-critical workloads."
                saving = max(saving, round(cost * 0.20, 2))

        if svc == "CloudWatch" and row["storage_gb"] > 100:
            issue = "Excessive Log Retention"
            rec = "Reduce log retention period and archive cold logs."
            saving = round(cost * 0.15, 2)

        if svc == "S3" and row["storage_gb"] > 2000:
            issue = "Storage Lifecycle Opportunity"
            rec = "Move old objects to lower-cost storage tiers and remove stale files."
            saving = round(cost * 0.18, 2)

        issue_type.append(issue)
        recommendations.append(rec)
        estimated_saving.append(saving)

    out["issue_type"] = issue_type
    out["recommendation"] = recommendations
    out["estimated_daily_saving_usd"] = estimated_saving
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("service", as_index=False)
        .agg(
            total_cost_usd=("daily_cost_usd", "sum"),
            avg_daily_cost_usd=("daily_cost_usd", "mean"),
            potential_saving_usd=("estimated_daily_saving_usd", "sum"),
        )
        .sort_values("total_cost_usd", ascending=False)
    )

    summary["avg_daily_cost_usd"] = summary["avg_daily_cost_usd"].round(2)
    summary["total_cost_usd"] = summary["total_cost_usd"].round(2)
    summary["potential_saving_usd"] = summary["potential_saving_usd"].round(2)
    return summary


def build_issue_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("issue_type", as_index=False)
        .agg(
            observations=("issue_type", "count"),
            potential_saving_usd=("estimated_daily_saving_usd", "sum"),
        )
        .sort_values("potential_saving_usd", ascending=False)
    )


def build_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "usage_date",
            "service",
            "environment",
            "daily_cost_usd",
            "issue_type",
            "recommendation",
            "estimated_daily_saving_usd",
        ]
    ].copy()

    out = out[out["issue_type"] != "Healthy"].sort_values(
        ["estimated_daily_saving_usd", "daily_cost_usd"],
        ascending=[False, False],
    )
    return out


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("💸 Cloud Cost Optimization Dashboard")
st.caption(
    "A data engineer-style dashboard for cloud spend monitoring, usage analysis, and optimization recommendations."
)

st.markdown(
    """
### What this project demonstrates
- cloud cost monitoring
- service-level spend analysis
- utilization-based optimization logic
- rightsizing recommendations
- warehouse scan-cost reduction ideas
- estimated savings reporting
"""
)

source_mode = st.radio(
    "Choose data source",
    ["Use sample cloud cost data", "Upload CSV"],
    horizontal=True,
)

if source_mode == "Use sample cloud cost data":
    raw_df = generate_cloud_cost_data(days=30)
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    raw_df = pd.read_csv(uploaded_file)

required_cols = ["usage_date", "service", "daily_cost_usd"]
missing_cols = [c for c in required_cols if c not in raw_df.columns]
if missing_cols:
    st.error(f"Missing required columns: {', '.join(missing_cols)}")
    st.stop()

raw_df["usage_date"] = pd.to_datetime(raw_df["usage_date"], errors="coerce")
raw_df["daily_cost_usd"] = pd.to_numeric(raw_df["daily_cost_usd"], errors="coerce").fillna(0)

optional_cols = [
    "environment",
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "storage_gb",
    "query_scan_gb",
    "idle_hours",
]

for col in optional_cols:
    if col not in raw_df.columns:
        raw_df[col] = None

services = sorted(raw_df["service"].dropna().astype(str).unique().tolist())
environments = sorted(raw_df["environment"].dropna().astype(str).unique().tolist()) if "environment" in raw_df.columns else []

with st.sidebar:
    st.header("Filters")
    selected_services = st.multiselect("Service", services, default=services)
    selected_envs = st.multiselect("Environment", environments, default=environments if environments else [])
    min_cost = st.slider(
        "Minimum Daily Cost",
        min_value=float(raw_df["daily_cost_usd"].min()),
        max_value=float(raw_df["daily_cost_usd"].max()) if len(raw_df) else 1.0,
        value=float(raw_df["daily_cost_usd"].min()),
    )

filtered_df = raw_df[raw_df["service"].isin(selected_services)].copy()
if selected_envs:
    filtered_df = filtered_df[filtered_df["environment"].isin(selected_envs)]
filtered_df = filtered_df[filtered_df["daily_cost_usd"] >= min_cost]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

optimized_df = classify_cost_issues(filtered_df)
summary_df = build_summary(optimized_df)
issue_df = build_issue_summary(optimized_df)
recommendation_df = build_recommendation_table(optimized_df)

total_cost = round(float(optimized_df["daily_cost_usd"].sum()), 2)
total_potential_saving = round(float(optimized_df["estimated_daily_saving_usd"].sum()), 2)
monthly_saving_projection = round(total_potential_saving * 30, 2)
high_opportunity_count = int((optimized_df["estimated_daily_saving_usd"] > 0).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cost", f"${total_cost:,.2f}")
k2.metric("Potential Daily Saving", f"${total_potential_saving:,.2f}")
k3.metric("Projected Monthly Saving", f"${monthly_saving_projection:,.2f}")
k4.metric("Optimization Opportunities", high_opportunity_count)

st.markdown("---")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("Service Cost Summary")
    render_html_table(summary_df, height=320)

with right:
    st.subheader("Issue Summary")
    render_html_table(issue_df, height=320)

st.markdown("---")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Total Cost by Service")
    if not summary_df.empty:
        cost_series = summary_df.set_index("service")["total_cost_usd"]
        render_bar_chart(cost_series, "Total Cost by Service", "Service", "USD")
    else:
        st.info("No cost summary available.")

with chart_right:
    st.subheader("Potential Savings by Issue Type")
    if not issue_df.empty:
        saving_series = issue_df.set_index("issue_type")["potential_saving_usd"]
        render_bar_chart(saving_series, "Potential Savings by Issue", "Issue Type", "USD")
    else:
        st.info("No issue summary available.")

st.markdown("---")

st.subheader("Daily Spend Trend")
daily_trend = (
    optimized_df.groupby(["usage_date", "service"])["daily_cost_usd"]
    .sum()
    .reset_index()
    .pivot(index="usage_date", columns="service", values="daily_cost_usd")
    .fillna(0)
    .sort_index()
)
render_line_chart(daily_trend, "Daily Spend Trend by Service", "Usage Date", "Daily Cost (USD)")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Optimization Recommendations", "Detailed Usage Data", "Engineering Notes"])

with tab1:
    st.subheader("Optimization Recommendations")
    if recommendation_df.empty:
        st.success("No optimization opportunities detected.")
    else:
        render_html_table(recommendation_df, height=420)

with tab2:
    st.subheader("Detailed Usage Data")
    display_df = optimized_df.copy()
    display_df["usage_date"] = display_df["usage_date"].dt.strftime("%Y-%m-%d")
    render_html_table(display_df.head(200), height=420)

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Detailed Analysis CSV",
        data=csv_data,
        file_name="cloud_cost_analysis.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Engineering Notes")
    st.markdown(
        """
**Checks included**
- underutilized compute detection
- high idle time detection
- high scan-cost warehouse detection
- excessive storage / retention opportunity
- service-level cost aggregation
- projected savings estimation

**Typical data engineering / platform use cases**
- cloud spend review meetings
- warehouse scan reduction planning
- EC2 / RDS / cluster rightsizing analysis
- infrastructure cost reporting to leadership
- optimization backlog prioritization

**Potential future upgrades**
- actual AWS / Azure / GCP billing export ingestion
- reservation / savings-plan simulation
- before-vs-after savings tracking
- per-team chargeback reporting
- anomaly detection on daily spend spikes
"""
    )