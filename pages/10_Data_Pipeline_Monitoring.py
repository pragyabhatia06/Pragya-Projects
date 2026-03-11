# app.py

import random
from datetime import datetime, timedelta

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
    running_count = (latest["status"] == "Running").sum()
    failed_count = (latest["status"] == "Failed").sum()
    success_rate = round((df["status"] == "Success").mean() * 100, 1)
    sla_breach_rate = round(df["sla_breached"].mean() * 100, 1)
    total_records = int(latest["record_count"].sum())

    return total_pipelines, running_count, failed_count, success_rate, sla_breach_rate, total_records


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
st.dataframe(snapshot_df, use_container_width=True)

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
    )
    st.line_chart(trend_df)

with right_col:
    st.subheader("SLA Breaches by Pipeline")
    breach_df = (
        filtered.groupby("pipeline_name")["sla_breached"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(breach_df)

left_col_2, right_col_2 = st.columns(2)

with left_col_2:
    st.subheader("Record Counts by Pipeline")
    record_df = (
        filtered.groupby("pipeline_name")["record_count"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(record_df)

with right_col_2:
    st.subheader("Average Runtime by Pipeline (mins)")
    runtime_df = (
        filtered.groupby("pipeline_name")["duration_minutes"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(runtime_df)

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
    st.dataframe(
        error_logs.rename(
            columns={
                "scheduled_time": "Run Time",
                "pipeline_name": "Pipeline",
                "status": "Status",
                "duration_minutes": "Duration (mins)",
                "error_message": "Error Message",
            }
        ),
        use_container_width=True,
    )

with st.expander("Raw Pipeline Run Data"):
    raw_df = filtered.copy()
    raw_df["scheduled_time"] = raw_df["scheduled_time"].apply(format_dt)
    raw_df["actual_start"] = raw_df["actual_start"].apply(format_dt)
    raw_df["completed_at"] = raw_df["completed_at"].apply(format_dt)
    st.dataframe(raw_df, use_container_width=True)