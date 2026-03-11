import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import hashlib
import json
import uuid
from datetime import datetime
import streamlit as st
from pathlib import Path


# -------------------------
# Config
# -------------------------

BASE_URL = "https://quotes.toscrape.com"
DB_FILE = "quotes_pipeline.db"
PIPELINE_LOG = "pipeline_runs.jsonl"


# -------------------------
# Scraping Layer
# -------------------------

def scrape_quotes():

    url = f"{BASE_URL}/page/1/"
    all_rows = []
    page_no = 1

    while url:

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        quotes = soup.select("div.quote")

        for q in quotes:

            text = q.select_one("span.text").text
            author = q.select_one("small.author").text
            tags = [t.text for t in q.select("div.tags a.tag")]

            all_rows.append({
                "quote_text": text,
                "author": author,
                "tags": ",".join(tags),
                "source_page": page_no,
                "scraped_at": datetime.utcnow().isoformat()
            })

        next_btn = soup.select_one("li.next a")

        if next_btn:
            url = BASE_URL + next_btn.get("href")
            page_no += 1
        else:
            url = None

    return pd.DataFrame(all_rows)


# -------------------------
# Transform Layer
# -------------------------

def transform_data(df):

    df["quote_text"] = df["quote_text"].str.strip()
    df["author"] = df["author"].str.strip()

    df["record_hash"] = df.apply(
        lambda r: hashlib.md5(
            f"{r['quote_text']}-{r['author']}".encode()
        ).hexdigest(),
        axis=1
    )

    df = df.drop_duplicates("record_hash")

    return df


# -------------------------
# Load Layer
# -------------------------

def load_to_db(df):

    conn = sqlite3.connect(DB_FILE)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS quotes (
        quote_text TEXT,
        author TEXT,
        tags TEXT,
        source_page INTEGER,
        scraped_at TEXT,
        record_hash TEXT UNIQUE
    )
    """)

    inserted = 0

    for _, row in df.iterrows():

        try:

            conn.execute(
                """
                INSERT INTO quotes
                VALUES (?,?,?,?,?,?)
                """,
                (
                    row["quote_text"],
                    row["author"],
                    row["tags"],
                    row["source_page"],
                    row["scraped_at"],
                    row["record_hash"]
                )
            )

            inserted += 1

        except:
            pass

    conn.commit()
    conn.close()

    return inserted


# -------------------------
# Pipeline Logging
# -------------------------

def log_pipeline(record):

    with open(PIPELINE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# -------------------------
# Run Pipeline
# -------------------------

def run_pipeline():

    run_id = str(uuid.uuid4())
    start = datetime.utcnow()

    try:

        raw_df = scrape_quotes()
        transformed = transform_data(raw_df)
        loaded = load_to_db(transformed)

        end = datetime.utcnow()

        record = {
            "run_id": run_id,
            "status": "SUCCESS",
            "records_extracted": len(raw_df),
            "records_loaded": loaded,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_sec": (end - start).total_seconds()
        }

        log_pipeline(record)

        return record

    except Exception as e:

        end = datetime.utcnow()

        record = {
            "run_id": run_id,
            "status": "FAILED",
            "error": str(e),
            "start_time": start.isoformat(),
            "end_time": end.isoformat()
        }

        log_pipeline(record)

        raise


# -------------------------
# Read Logs
# -------------------------

def read_logs():

    if not Path(PIPELINE_LOG).exists():
        return pd.DataFrame()

    rows = []

    with open(PIPELINE_LOG) as f:

        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


# -------------------------
# Read Data
# -------------------------

def read_quotes():

    if not Path(DB_FILE).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM quotes", conn)
    conn.close()

    return df


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Web Scraping Data Pipeline", layout="wide")

st.title("Web Scraping Data Engineering Pipeline")

st.write("This app demonstrates an end-to-end data pipeline: scrape → transform → load → monitor.")

if st.button("Run Scraping Pipeline"):

    with st.spinner("Running pipeline..."):

        result = run_pipeline()

    st.success("Pipeline completed")

    st.json(result)


st.header("Pipeline Monitoring")

logs = read_logs()

if not logs.empty:

    col1, col2, col3 = st.columns(3)

    col1.metric("Pipeline Runs", len(logs))
    col2.metric("Success Runs", (logs["status"] == "SUCCESS").sum())
    col3.metric("Failed Runs", (logs["status"] == "FAILED").sum())

    st.line_chart(logs["duration_sec"])

    st.dataframe(logs)

else:
    st.info("No pipeline runs yet")


st.header("Loaded Quotes")

quotes = read_quotes()

if not quotes.empty:

    st.dataframe(quotes)

else:
    st.info("No data loaded yet")