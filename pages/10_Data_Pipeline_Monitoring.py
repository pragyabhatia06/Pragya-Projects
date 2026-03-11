# app.py

import random
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Data Pipeline Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def generate_pipeline_runs(days: int = 30) -> pd.DataFrame:
    pipelines = [
        {"pipeline_name": "customer_ingestion", "owner": "Data Platform", "sla_minutes": 30, "frequency": "Hourly"},
        {"pipeline_name": "orders_etl", "owner": "Commerce", "sla_minutes": 45, "frequency": "Hourly"},
        {"pipeline_name": "inventory_sync", "owner": "Supply Chain", "sla_minutes": 60, "frequency": "Daily"},
        {"pipeline_name": "finance_reporting", "owner": "Finance", "sla_minutes": 90, "frequency": "Daily"},
        {"pipeline_name": "marketing_attribution", "owner": "Marketing", "sla_minutes": 120, "frequency": "Daily"},
        {"pipeline_name": "crm_snapshot", "owner": "Sales", "sla_minutes": 50, "frequency": "Daily"},
    ]

    rows = []
    now = datetime.now().replace(second=0, microsecond=0)

    for pipeline in pipelines:
        runs = days * (12 if pipeline["frequency"] == "Hourly" else 1)
        run_time = now - timedelta(days=days)

        for _ in range(runs):
            run_time += timedelta(hours=2 if pipeline["frequency"] == "Hourly" else 24)

            scheduled_time = run_time
            duration = max(5, int(random.gauss(mu=pipeline["sla_minutes"] * 0.75, sigma=10)))

            status = random.choices(
                ["Success", "Failed", "Running", "Delayed"],
                weights=[78, 10, 4, 8],
                k=1,
            )[0]

            if status == "Failed":
                duration = min(duration, pipeline["sla_minutes"] + random.randint(5, 30))
                record_count = random.randint(0, 1000)
                error_message = random.choice(
                    [
                        "Connection timeout while reading source API",
                        "Primary key violation on target table",
                        "S3 object missing for expected partition",
                        "Schema drift detected in upstream payload",
                        "Warehouse lock timeout during merge step",
                    ]
                )
            elif status == "Running":
                duration = random.randint(1, pipeline["sla_minutes"])
                record_count = random.randint(10000, 200000)
                error_message = None
            else:
                record_count = random.randint(5000, 300000)
                error_message = None

            if status == "Delayed":
                error_message = "Pipeline started late due to upstream dependency delay"

            actual_start = scheduled_time + timedelta(minutes=random.randint(0, 15))
            completed_at = None if status == "Running" else actual_start + timedelta(minutes=duration)
            sla_breached = duration > pipeline["sla_minutes"] or status in ["Failed", "Delayed"]

            rows.append(
                {
                    "pipeline_name": pipeline["pipeline_name"],
                    "owner": pipeline["owner"],
                    "frequency": pipeline["frequency"],
                    "scheduled_time": scheduled_time,
                    "actual_start": actual_start,
                    "completed_at": completed_at,
                    "duration_minutes": duration,
                    "sla_minutes": pipeline["sla_minutes"],
                    "status": status,
                    "record_count": record_count,
                    "sla_breached": sla_breached,
                    "error_message": error_message,
                }
            )

    df = pd.DataFrame(rows).sort_values(
        ["pipeline_name", "scheduled_time"],
        ascending=[True, False],
    )
    return df


def format_dt(value):
    if pd.isna(value):
        return "—"
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def latest_pipeline_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest = (
        df.sort_values("scheduled_time", ascending=False)
        .groupby("pipeline_name", as_index=False)
        .first()
        .sort_values("pipeline_name")
    )

    summary_rows = []

    for pipeline_name in latest["pipeline_name"]:
        pipeline_df = df[df["pipeline_name"] == pipeline_name]
        latest_row = pipeline_df.sort_values("scheduled_time", ascending=False).iloc[0]

        success_rate = round((pipeline_df["status"] == "Success").mean() * 100, 1)
        failure_rate = round((pipeline_df["status"] == "Failed").mean() * 100, 1)
        sla_breach_rate = round(pipeline_df["sla_breached"].mean() * 100, 1)

        summary_rows.append(
            {
                "Pipeline": pipeline_name,
                "Owner": latest_row["owner"],
                "Frequency": latest_row["frequency"],
                "Current Status": latest_row["status"],
                "Last Run Time": format_dt(latest_row["scheduled_time"]),
                "Duration (mins)": int(latest_row["duration_minutes"]),
                "SLA (mins)": int(latest_row["sla_minutes"]),
                "Success Rate %": success_rate,
                "Failure Rate %": failure_rate,
                "SLA Breach %": sla_breach_rate,
                "Last Record Count": int(latest_row["record_count"]),
                "Latest Error": latest_row["error_message"] or "—",
            }
        )

    return pd.DataFrame(summary_rows)


def build_kpis(df: pd.DataFrame):
    latest = (
        df.sort_values("scheduled_time", ascending=False)
        .groupby("pipeline_name", as_index=False)
        .first()
    )

    total_pipelines = len(latest)
    running_count = int((latest["status"] == "Running").sum())
    failed_count = int((latest["status"] == "Failed").sum())
    success_rate = round((df["status"] == "Success").mean() * 100, 1) if not df.empty else 0
    sla_breach_rate = round(df["sla_breached"].mean() * 100, 1) if not df.empty else 0
    total_records = int(latest["record_count"].sum()) if not latest.empty else 0

    return total_pipelines, running_count, failed_count, success_rate, sla_breach_rate, total_records


def sanitize_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()

    for col in safe_df.columns:
        safe_df[col] = safe_df[col].apply(
            lambda x: "" if pd.isna(x) else str(x)
        )

    return safe_df


def render_html_table(df: pd.DataFrame, max_height_px: int = 420) -> None:
    if df.empty:
        st.info("No data available.")
        return

    safe_df = sanitize_for_streamlit(df).copy()
    html_table = safe_df.to_html(index=False, escape=True)

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

    st.markdown(
        f"""
        <div style="overflow:auto; max-height:{max_height_px}px; border:1px solid #ddd; border-radius:8px; padding:6px; background:#fff;">
            {html_table}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_line_chart(trend_df: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    if trend_df.empty:
        st.info("No chart data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    for col in trend_df.columns:
        ax.plot(trend_df.index.astype(str), trend_df[col], marker="o", label=str(col))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


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


st.title("📊 Data Pipeline Monitoring Dashboard")
st.caption(
    "Production-style monitoring dashboard for ETL / ELT pipeline health, last run visibility, SLA adherence, and error tracking."
)

df = generate_pipeline_runs(days=30)

with st.sidebar:
    st.header("Filters")

    selected_pipelines = st.multiselect(
        "Pipeline",
        sorted(df["pipeline_name"].unique()),
        default=sorted(df["pipeline_name"].unique()),
    )

    selected_statuses = st.multiselect(
        "Status",
        ["Success", "Failed", "Running", "Delayed"],
        default=["Success", "Failed", "Running", "Delayed"],
    )

    selected_owners = st.multiselect(
        "Owner",
        sorted(df["owner"].unique()),
        default=sorted(df["owner"].unique()),
    )

    last_n_days = st.slider("Last N days", 1, 30, 14)

filtered = df[
    (df["pipeline_name"].isin(selected_pipelines))
    & (df["status"].isin(selected_statuses))
    & (df["owner"].isin(selected_owners))
    & (df["scheduled_time"] >= (datetime.now() - timedelta(days=last_n_days)))
].copy()

total_pipelines, running_count, failed_count, success_rate, sla_breach_rate, total_records = build_kpis(filtered)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Pipelines", total_pipelines)
col2.metric("Running", running_count)
col3.metric("Failed", failed_count)
col4.metric("Success Rate", f"{success_rate}%")
col5.metric("SLA Breach Rate", f"{sla_breach_rate}%")
col6.metric("Latest Records", f"{total_records:,}")

st.markdown("---")

st.subheader("Current Pipeline Status")
snapshot_df = latest_pipeline_snapshot(filtered)
render_html_table(snapshot_df, max_height_px=360)

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Run Trend by Status")
    trend_df = (
        filtered.assign(run_date=filtered["scheduled_time"].dt.date)
        .groupby(["run_date", "status"])
        .size()
        .reset_index(name="runs")
        .pivot(index="run_date", columns="status", values="runs")
        .fillna(0)
        .sort_index()
    )
    render_line_chart(
        trend_df,
        title="Run Trend by Status",
        xlabel="Run Date",
        ylabel="Runs",
    )

with right_col:
    st.subheader("SLA Breaches by Pipeline")
    breach_df = (
        filtered.groupby("pipeline_name")["sla_breached"]
        .sum()
        .sort_values(ascending=False)
    )
    render_bar_chart(
        breach_df,
        title="SLA Breaches by Pipeline",
        xlabel="Pipeline",
        ylabel="Breach Count",
    )

left_col_2, right_col_2 = st.columns(2)

with left_col_2:
    st.subheader("Record Counts by Pipeline")
    record_df = (
        filtered.groupby("pipeline_name")["record_count"]
        .sum()
        .sort_values(ascending=False)
    )
    render_bar_chart(
        record_df,
        title="Record Counts by Pipeline",
        xlabel="Pipeline",
        ylabel="Records",
    )

with right_col_2:
    st.subheader("Average Runtime by Pipeline (mins)")
    runtime_df = (
        filtered.groupby("pipeline_name")["duration_minutes"]
        .mean()
        .sort_values(ascending=False)
    )
    render_bar_chart(
        runtime_df,
        title="Average Runtime by Pipeline",
        xlabel="Pipeline",
        ylabel="Minutes",
    )

st.markdown("---")

st.subheader("Recent Error Logs")
error_logs = filtered[filtered["error_message"].notna()].copy()
error_logs = error_logs.sort_values("scheduled_time", ascending=False)[
    ["scheduled_time", "pipeline_name", "status", "duration_minutes", "error_message"]
]
error_logs["scheduled_time"] = error_logs["scheduled_time"].apply(format_dt)

if error_logs.empty:
    st.success("No error logs found for the selected filters.")
else:
    display_error_logs = error_logs.rename(
        columns={
            "scheduled_time": "Run Time",
            "pipeline_name": "Pipeline",
            "status": "Status",
            "duration_minutes": "Duration (mins)",
            "error_message": "Error Message",
        }
    )
    render_html_table(display_error_logs, max_height_px=360)

with st.expander("Raw Pipeline Run Data"):
    raw_df = filtered.copy()
    raw_df["scheduled_time"] = raw_df["scheduled_time"].apply(format_dt)
    raw_df["actual_start"] = raw_df["actual_start"].apply(format_dt)
    raw_df["completed_at"] = raw_df["completed_at"].apply(format_dt)
    render_html_table(raw_df, max_height_px=500)