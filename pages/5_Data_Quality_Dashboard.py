import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Data Quality & Validation Dashboard",
    page_icon="✅",
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


# ---------------------------------------------------
# Sample data
# ---------------------------------------------------
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    data = {
        "customer_id": [101, 102, 103, 103, 105, None, 107, 108],
        "customer_name": [
            "Alice Johnson",
            "Bob Smith",
            "Charlie Brown",
            "Charlie Brown",
            None,
            "Eva Green",
            "Frank Hall",
            "Grace Lee",
        ],
        "email": [
            "alice@example.com",
            "bob@example",
            "charlie@example.com",
            "charlie@example.com",
            "eva@example.com",
            None,
            "frank@example.com",
            "grace@example.com",
        ],
        "signup_date": [
            "2024-01-10",
            "2024-02-15",
            "invalid_date",
            "2024-03-01",
            "2024-03-05",
            "2024-04-10",
            None,
            "2024-05-01",
        ],
        "country": ["UK", "UK", "US", "US", "IN", "UK", None, "FR"],
        "monthly_spend": [120.5, -10, 300.0, 300.0, None, 220.0, 150.5, 99999],
        "is_active": ["Yes", "No", "Yes", "Yes", "No", "Yes", "Invalid", "Yes"],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------
# Data quality functions
# ---------------------------------------------------
def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "row_count": len(df),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round((df[col].isna().sum() / len(df)) * 100, 2) if len(df) else 0,
                "unique_count": int(df[col].nunique(dropna=True)),
                "sample_value": safe_str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else "",
            }
        )

    return pd.DataFrame(rows)


def find_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    dup_mask = df.duplicated(keep=False)
    out = df[dup_mask].copy()
    if not out.empty:
        out.insert(0, "duplicate_flag", "Y")
    return out


def validate_email_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    out = df.copy()
    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    def check_email(val):
        if pd.isna(val):
            return "NULL"
        return "VALID" if re.match(email_pattern, str(val).strip()) else "INVALID"

    out["email_validation_status"] = out[column_name].apply(check_email)
    return out


def validate_date_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    out = df.copy()
    parsed = pd.to_datetime(out[column_name], errors="coerce")
    out["date_validation_status"] = parsed.apply(lambda x: "VALID" if pd.notna(x) else "INVALID")
    return out


def validate_numeric_range(df: pd.DataFrame, column_name: str, min_value=None, max_value=None) -> pd.DataFrame:
    out = df.copy()
    numeric_series = pd.to_numeric(out[column_name], errors="coerce")

    def check_range(val):
        if pd.isna(val):
            return "NULL"
        if min_value is not None and val < min_value:
            return "OUT_OF_RANGE"
        if max_value is not None and val > max_value:
            return "OUT_OF_RANGE"
        return "VALID"

    out[f"{column_name}_range_status"] = numeric_series.apply(check_range)
    return out


def validate_allowed_values(df: pd.DataFrame, column_name: str, allowed_values: list[str]) -> pd.DataFrame:
    out = df.copy()

    def check_value(val):
        if pd.isna(val):
            return "NULL"
        return "VALID" if str(val).strip() in allowed_values else "INVALID"

    out[f"{column_name}_allowed_status"] = out[column_name].apply(check_value)
    return out


def build_issue_summary(
    df: pd.DataFrame,
    email_col: str | None,
    date_col: str | None,
    numeric_col: str | None,
    allowed_col: str | None,
    allowed_values: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    issues = []

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    issues.append({"check_name": "Duplicate Rows", "issue_count": duplicate_count})

    # Email validation
    if email_col and email_col in working.columns:
        working = validate_email_column(working, email_col)
        invalid_email_count = int((working["email_validation_status"] == "INVALID").sum())
        issues.append({"check_name": f"Invalid Emails ({email_col})", "issue_count": invalid_email_count})

    # Date validation
    if date_col and date_col in working.columns:
        working = validate_date_column(working, date_col)
        invalid_date_count = int((working["date_validation_status"] == "INVALID").sum())
        issues.append({"check_name": f"Invalid Dates ({date_col})", "issue_count": invalid_date_count})

    # Numeric validation
    if numeric_col and numeric_col in working.columns:
        working = validate_numeric_range(working, numeric_col, min_value=0, max_value=10000)
        invalid_numeric_count = int((working[f"{numeric_col}_range_status"] == "OUT_OF_RANGE").sum())
        issues.append({"check_name": f"Out of Range Values ({numeric_col})", "issue_count": invalid_numeric_count})

    # Allowed values validation
    if allowed_col and allowed_col in working.columns and allowed_values:
        working = validate_allowed_values(working, allowed_col, allowed_values)
        invalid_allowed_count = int((working[f"{allowed_col}_allowed_status"] == "INVALID").sum())
        issues.append({"check_name": f"Invalid Allowed Values ({allowed_col})", "issue_count": invalid_allowed_count})

    issue_df = pd.DataFrame(issues)
    return working, issue_df


def create_cleaned_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col.endswith("_status"):
            continue

    # Remove fully duplicated rows
    out = out.drop_duplicates().reset_index(drop=True)

    # Standardize object columns
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip()
            out[col] = out[col].replace({"None": None, "nan": None, "": None})

    return out


def render_bar_chart(issue_df: pd.DataFrame):
    if issue_df.empty:
        st.info("No issue summary available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(issue_df["check_name"], issue_df["issue_count"])
    ax.set_title("Data Quality Issues by Check")
    ax.set_xlabel("Check")
    ax.set_ylabel("Issue Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("✅ Data Quality & Validation Dashboard")
st.caption("A data engineer-style quality control app for profiling, validation, anomaly detection, and cleaned export.")

st.markdown(
    """
### What this project demonstrates
- dataset ingestion
- schema profiling
- null analysis
- duplicate detection
- email/date/type validation
- allowed-value checks
- anomaly flagging
- cleaned export generation
"""
)

source_option = st.radio(
    "Choose data source",
    ["Use sample dataset", "Upload CSV"],
    horizontal=True,
)

if source_option == "Use sample dataset":
    df = load_sample_data()
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Upload a CSV to continue.")
        st.stop()

if df.empty:
    st.warning("No data available.")
    st.stop()

st.markdown("---")

# Dataset overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Null Cells", int(df.isna().sum().sum()))
c4.metric("Duplicate Rows", int(df.duplicated().sum()))

st.markdown("---")

with st.expander("Raw Dataset Preview", expanded=True):
    render_html_table(df.head(50), height=320)

# Validation setup
st.subheader("Validation Configuration")

config_col1, config_col2, config_col3, config_col4 = st.columns(4)

with config_col1:
    email_col = st.selectbox("Email Column", [""] + df.columns.tolist())

with config_col2:
    date_col = st.selectbox("Date Column", [""] + df.columns.tolist())

with config_col3:
    numeric_col = st.selectbox("Numeric Range Column", [""] + df.columns.tolist())

with config_col4:
    allowed_col = st.selectbox("Allowed Values Column", [""] + df.columns.tolist())

allowed_values_input = ""
if allowed_col:
    allowed_values_input = st.text_input(
        "Allowed Values (comma-separated)",
        value="Yes,No",
    )

allowed_values = [v.strip() for v in allowed_values_input.split(",") if v.strip()] if allowed_values_input else []

validated_df, issue_df = build_issue_summary(
    df=df,
    email_col=email_col if email_col else None,
    date_col=date_col if date_col else None,
    numeric_col=numeric_col if numeric_col else None,
    allowed_col=allowed_col if allowed_col else None,
    allowed_values=allowed_values if allowed_values else None,
)

profile_df = build_column_profile(df)
duplicates_df = find_duplicate_rows(df)
cleaned_df = create_cleaned_export(validated_df)

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Column Profile")
    render_html_table(profile_df, height=320)

with right:
    st.subheader("Issue Summary")
    render_html_table(issue_df, height=320)

st.markdown("---")

chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.subheader("Issue Distribution")
    render_bar_chart(issue_df)

with chart_col2:
    st.subheader("Validation Summary")
    total_issues = int(issue_df["issue_count"].sum()) if not issue_df.empty else 0
    st.metric("Total Quality Issues", total_issues)
    st.metric("Execution Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.metric("Cleaned Rows", len(cleaned_df))

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Validated Dataset", "Duplicate Rows", "Cleaned Export"])

with tab1:
    st.subheader("Validated Dataset")
    render_html_table(validated_df.head(100), height=400)

with tab2:
    st.subheader("Duplicate Rows")
    if duplicates_df.empty:
        st.success("No duplicate rows found.")
    else:
        render_html_table(duplicates_df, height=320)

with tab3:
    st.subheader("Cleaned Dataset")
    render_html_table(cleaned_df.head(100), height=400)

    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

st.markdown("---")

with st.expander("Engineering Notes"):
    st.markdown(
        """
**Source layer**
- CSV upload or sample dataset ingestion

**Profiling layer**
- row/column counts
- null counts
- unique counts
- dtype inspection

**Validation layer**
- email format checks
- date parsing checks
- numeric range checks
- allowed value checks
- duplicate row detection

**Output layer**
- validated dataset preview
- issue summary
- cleaned export for downstream use

**Business relevance**
- useful for ingestion checks, bronze-to-silver validation, API payload quality checks, and pre-warehouse screening
"""
    )