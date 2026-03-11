from urllib.error import URLError

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Agricultural Data Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------
# Safe display helpers
# -----------------------------------
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


def render_html_table(df: pd.DataFrame, height: int = 420):
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


def render_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    if series.empty:
        st.info("No chart data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(series.index.astype(str), series.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    st.pyplot(fig)


# -----------------------------------
# Data ingestion
# -----------------------------------
@st.cache_data
def load_source_data() -> pd.DataFrame:
    aws_bucket_url = "https://streamlit-demo-data.s3-us-west-2.amazonaws.com"
    df = pd.read_csv(aws_bucket_url + "/agri.csv.gz")
    return df


# -----------------------------------
# Transformations
# -----------------------------------
def transform_source_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out.columns = [str(col).strip() for col in out.columns]
    out["Region"] = out["Region"].astype(str).str.strip()

    year_columns = [col for col in out.columns if col != "Region"]

    for col in year_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def build_country_timeseries(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    filtered = df[df["Region"].isin(countries)].copy()

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.set_index("Region")
    filtered = filtered.loc[countries]
    filtered = filtered / 1_000_000_000  # convert to billions
    return filtered.T


def build_latest_snapshot(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    filtered = df[df["Region"].isin(countries)].copy()
    if filtered.empty:
        return pd.DataFrame()

    year_columns = [col for col in filtered.columns if col != "Region"]
    latest_year = sorted(year_columns)[-1]

    snapshot = filtered[["Region", latest_year]].copy()
    snapshot = snapshot.rename(columns={latest_year: "Latest Gross Agricultural Production"})
    snapshot["Latest Gross Agricultural Production"] = (
        snapshot["Latest Gross Agricultural Production"] / 1_000_000_000
    ).round(2)

    snapshot = snapshot.sort_values("Latest Gross Agricultural Production", ascending=False)
    return snapshot


def build_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "non_null_count": int(df[col].notna().sum()),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------
# UI
# -----------------------------------
st.title("📊 Agricultural Data Pipeline & Analytics Dashboard")
st.caption(
    "A data engineering-style dashboard showing source ingestion, transformation, data quality checks, and analytics."
)

st.markdown(
    """
### What this project demonstrates
- external dataset ingestion
- cached source loading
- transformation into analytics-ready format
- KPI calculation
- basic data quality reporting
- filtered trend analysis
- dashboard presentation for business users
"""
)

try:
    raw_df = load_source_data()
    curated_df = transform_source_data(raw_df)

    available_countries = sorted(curated_df["Region"].dropna().unique().tolist())

    with st.sidebar:
        st.header("Filters")
        selected_countries = st.multiselect(
            "Choose countries",
            available_countries,
            default=["China", "United States of America"] if "China" in available_countries and "United States of America" in available_countries else available_countries[:2],
        )

    if not selected_countries:
        st.error("Please select at least one country.")
        st.stop()

    timeseries_df = build_country_timeseries(curated_df, selected_countries)
    latest_snapshot_df = build_latest_snapshot(curated_df, selected_countries)
    dq_df = build_data_quality_report(curated_df)

    # KPIs
    total_countries = curated_df["Region"].nunique()
    total_columns = len(curated_df.columns)
    latest_year = sorted([col for col in curated_df.columns if col != "Region"])[-1]
    selected_latest_total = (
        latest_snapshot_df["Latest Gross Agricultural Production"].sum()
        if not latest_snapshot_df.empty
        else 0
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Source Records", len(curated_df))
    k2.metric("Countries", total_countries)
    k3.metric("Available Years", total_columns - 1)
    k4.metric(f"Selected Total ({latest_year})", f"${selected_latest_total:.2f}B")

    st.markdown("---")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Latest Snapshot by Country")
        render_html_table(latest_snapshot_df, height=320)

    with right:
        st.subheader("Top Selected Countries by Latest Production")
        if not latest_snapshot_df.empty:
            bar_series = latest_snapshot_df.set_index("Region")["Latest Gross Agricultural Production"]
            render_bar_chart(
                bar_series,
                title=f"Latest Gross Agricultural Production ({latest_year})",
                xlabel="Country",
                ylabel="USD Billions",
            )
        else:
            st.info("No latest snapshot available.")

    st.markdown("---")

    st.subheader("Historical Trend Analysis")
    render_line_chart(
        timeseries_df,
        title="Gross Agricultural Production Trend",
        xlabel="Year",
        ylabel="USD Billions",
    )

    st.markdown("---")

    dq_left, dq_right = st.columns([1.2, 1])

    with dq_left:
        st.subheader("Curated Dataset Preview")
        render_html_table(curated_df.head(20), height=320)

    with dq_right:
        st.subheader("Data Quality Report")
        render_html_table(dq_df, height=320)

    st.markdown("---")

    with st.expander("Transformation Notes"):
        st.markdown(
            f"""
**Source layer**
- Raw file loaded from an external cloud-hosted CSV source

**Transformation layer**
- Standardized column names
- Converted year columns to numeric
- Prepared country-level time series for analytics

**Business layer**
- Latest snapshot by selected country
- Historical production trend
- Selected-country aggregate KPI

**Current unit**
- Gross Agricultural Production shown in **USD Billions**
- Latest year detected from source: **{latest_year}**
"""
        )

except URLError as e:
    st.error(
        f"""
This dashboard requires internet access to retrieve the source dataset.

Connection error: {e.reason}
"""
    )
except Exception as e:
    st.error(f"Application error: {e}")