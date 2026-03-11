import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


# =====================================
# CONFIG
# =====================================

BASE_URL = "https://quotes.toscrape.com"
DB_PATH = Path("quotes_pipeline.db")
PIPELINE_LOG_PATH = Path("pipeline_runs.jsonl")
REQUEST_TIMEOUT = 20


# =====================================
# HELPERS
# =====================================

def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value)


def make_streamlit_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all columns to plain string-compatible values
    to avoid Streamlit Arrow/LargeUtf8 rendering errors.
    """
    if df.empty:
        return df.copy()

    safe_df = df.copy()

    for col in safe_df.columns:
        try:
            safe_df[col] = safe_df[col].apply(safe_str)
        except Exception:
            safe_df[col] = safe_df[col].astype(str)

    return safe_df


def show_safe_dataframe(df: pd.DataFrame):
    if df.empty:
        st.info("No data available.")
        return

    st.dataframe(make_streamlit_safe(df), use_container_width=True, hide_index=True)


# =====================================
# DATABASE
# =====================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quotes (
                quote_id INTEGER PRIMARY KEY AUTOINCREMENT,
                quote_text TEXT NOT NULL,
                author TEXT NOT NULL,
                tags TEXT,
                source_page INTEGER,
                scraped_at TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                pipeline_name TEXT NOT NULL,
                status TEXT NOT NULL,
                records_extracted INTEGER NOT NULL,
                records_loaded INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_seconds REAL,
                error_message TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# =====================================
# SCRAPING
# =====================================

def scrape_quotes():
    url = f"{BASE_URL}/page/1/"
    all_rows = []
    page_no = 1

    while url:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        quote_blocks = soup.select("div.quote")

        for block in quote_blocks:
            text = block.select_one("span.text").get_text(strip=True)
            author = block.select_one("small.author").get_text(strip=True)
            tags = [tag.get_text(strip=True) for tag in block.select("div.tags a.tag")]

            all_rows.append(
                {
                    "quote_text": text,
                    "author": author,
                    "tags": ", ".join(tags),
                    "source_page": page_no,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        next_button = soup.select_one("li.next a")
        if next_button:
            next_href = next_button.get("href")
            url = f"{BASE_URL}{next_href}"
            page_no += 1
        else:
            url = None

    return pd.DataFrame(all_rows)


# =====================================
# TRANSFORM
# =====================================

def transform_quotes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["quote_text"] = out["quote_text"].astype(str).str.strip()
    out["author"] = out["author"].astype(str).str.strip()
    out["tags"] = out["tags"].astype(str).str.strip()
    out["source_page"] = pd.to_numeric(out["source_page"], errors="coerce").fillna(0).astype(int)

    out["record_hash"] = out.apply(
        lambda row: hashlib.md5(
            f"{row['quote_text']}|{row['author']}".encode("utf-8")
        ).hexdigest(),
        axis=1,
    )

    out = out.drop_duplicates(subset=["record_hash"]).reset_index(drop=True)
    return out


# =====================================
# LOAD
# =====================================

def load_quotes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)
    loaded = 0

    try:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT OR IGNORE INTO quotes (
                    quote_text,
                    author,
                    tags,
                    source_page,
                    scraped_at,
                    record_hash
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["quote_text"],
                    row["author"],
                    row["tags"],
                    int(row["source_page"]),
                    row["scraped_at"],
                    row["record_hash"],
                ),
            )
            loaded += cursor.rowcount

        conn.commit()
    finally:
        conn.close()

    return loaded


# =====================================
# PIPELINE LOGGING
# =====================================

def append_pipeline_log(record: dict):
    with open(PIPELINE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def insert_pipeline_run_db(record: dict):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs (
                run_id,
                pipeline_name,
                status,
                records_extracted,
                records_loaded,
                started_at,
                completed_at,
                duration_seconds,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("run_id"),
                record.get("pipeline_name"),
                record.get("status"),
                record.get("records_extracted", 0),
                record.get("records_loaded", 0),
                record.get("started_at"),
                record.get("completed_at"),
                record.get("duration_seconds"),
                record.get("error_message"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# =====================================
# PIPELINE
# =====================================

def run_pipeline():
    init_db()

    run_id = str(uuid.uuid4())
    pipeline_name = "quotes_web_scraping_pipeline"
    started_at = datetime.now(timezone.utc)

    try:
        raw_df = scrape_quotes()
        curated_df = transform_quotes(raw_df)
        loaded_count = load_quotes(curated_df)

        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()

        record = {
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "status": "SUCCESS",
            "records_extracted": int(len(raw_df)),
            "records_loaded": int(loaded_count),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration_seconds,
            "error_message": None,
        }

        append_pipeline_log(record)
        insert_pipeline_run_db(record)

        return record, curated_df

    except Exception as e:
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()

        record = {
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "status": "FAILED",
            "records_extracted": 0,
            "records_loaded": 0,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration_seconds,
            "error_message": str(e),
        }

        append_pipeline_log(record)
        insert_pipeline_run_db(record)
        raise


# =====================================
# READ FUNCTIONS
# =====================================

@st.cache_data
def read_pipeline_logs_file() -> pd.DataFrame:
    if not PIPELINE_LOG_PATH.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "pipeline_name",
                "status",
                "records_extracted",
                "records_loaded",
                "started_at",
                "completed_at",
                "duration_seconds",
                "error_message",
            ]
        )

    rows = []
    with open(PIPELINE_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows)


@st.cache_data
def read_pipeline_runs_db() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM pipeline_runs ORDER BY started_at DESC",
            conn,
        )
    finally:
        conn.close()

    return df


@st.cache_data
def read_quotes_db() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM quotes ORDER BY quote_id DESC",
            conn,
        )
    finally:
        conn.close()

    return df


# =====================================
# STREAMLIT UI
# =====================================

st.set_page_config(
    page_title="Web Scraping Data Engineering Pipeline",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Web Scraping Data Engineering Pipeline")
st.caption("End-to-end demo: scrape → transform → load → monitor")

st.markdown(
    """
This app demonstrates a simple production-style data engineering workflow:

- scrape data from a public website
- clean and deduplicate records
- load into SQLite
- log pipeline runs
- monitor status and loaded records
"""
)

init_db()

col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("Run Scraping Pipeline", use_container_width=True):
        try:
            with st.spinner("Running pipeline..."):
                result, curated_df = run_pipeline()

            st.success("Pipeline completed successfully.")
            st.json(result)

            st.subheader("Latest Curated Data")
            show_safe_dataframe(curated_df.head(20))

            st.cache_data.clear()

        except Exception as e:
            st.error(f"Pipeline failed: {e}")

with col_b:
    st.info(f"Database file: {DB_PATH.resolve()}")

st.markdown("---")

logs_df = read_pipeline_runs_db()
if logs_df.empty:
    logs_df = read_pipeline_logs_file()

quotes_df = read_quotes_db()

runs_count = 0 if logs_df.empty else len(logs_df)
success_count = 0 if logs_df.empty else int((logs_df["status"] == "SUCCESS").sum())
failed_count = 0 if logs_df.empty else int((logs_df["status"] == "FAILED").sum())
quotes_count = 0 if quotes_df.empty else len(quotes_df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pipeline Runs", runs_count)
k2.metric("Successful Runs", success_count)
k3.metric("Failed Runs", failed_count)
k4.metric("Quotes Loaded", quotes_count)

st.markdown("---")

left, right = st.columns(2)

with left:
    st.subheader("Pipeline Run History")
    if logs_df.empty:
        st.info("No pipeline runs yet.")
    else:
        show_safe_dataframe(logs_df)

with right:
    st.subheader("Run Duration Trend")
    if logs_df.empty or "duration_seconds" not in logs_df.columns:
        st.info("No duration data available.")
    else:
        chart_df = logs_df.copy()
        if "started_at" in chart_df.columns:
            chart_df["started_at"] = pd.to_datetime(chart_df["started_at"], errors="coerce")
            chart_df = chart_df.sort_values("started_at")
            chart_df = chart_df.dropna(subset=["started_at"])
            if not chart_df.empty:
                st.line_chart(chart_df.set_index("started_at")["duration_seconds"])
            else:
                st.info("No valid timestamp data available.")
        else:
            st.info("No started_at column available.")

st.markdown("---")

st.subheader("Loaded Quotes")
if quotes_df.empty:
    st.info("No quotes loaded yet. Run the pipeline first.")
else:
    show_safe_dataframe(quotes_df)

st.markdown("---")

with st.expander("Preview raw SQLite tables"):
    preview_choice = st.selectbox("Choose table", ["quotes", "pipeline_runs"])

    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            preview_df = pd.read_sql_query(f"SELECT * FROM {preview_choice} LIMIT 50", conn)
        finally:
            conn.close()

        show_safe_dataframe(preview_df)
    else:
        st.info("Database not found yet.")