# pages/2_Translator.py

from datetime import datetime
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Translation Pipeline Demo", page_icon="🌍", layout="wide")


# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
LOG_FILE = Path("translation_audit_log.jsonl")


# -------------------------------
# Safe display helpers
# -------------------------------
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


def render_html_table(df: pd.DataFrame, height: int = 360):
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


def render_request_trend(audit_df: pd.DataFrame):
    if audit_df.empty or "timestamp" not in audit_df.columns or "status" not in audit_df.columns:
        st.info("No trend data available.")
        return

    chart_df = audit_df.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
    chart_df = chart_df.dropna(subset=["timestamp"])

    if chart_df.empty:
        st.info("No valid timestamp data available.")
        return

    trend_df = (
        chart_df.assign(request_date=chart_df["timestamp"].dt.date)
        .groupby(["request_date", "status"])
        .size()
        .reset_index(name="count")
        .pivot(index="request_date", columns="status", values="count")
        .fillna(0)
        .sort_index()
    )

    if trend_df.empty:
        st.info("No trend data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for col in trend_df.columns:
        ax.plot(trend_df.index.astype(str), trend_df[col], marker="o", label=col)

    ax.set_title("Translation Request Trend")
    ax.set_xlabel("Request Date")
    ax.set_ylabel("Request Count")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


# -------------------------------
# Dependency-safe model loading
# -------------------------------
@st.cache_resource
def load_translation_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as e:
        raise ImportError(
            "Required package 'transformers' is not installed. Add it to requirements.txt."
        ) from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(
            f"Model loading failed for '{MODEL_NAME}'. Check whether 'sentencepiece', "
            "'torch', and model download access are available."
        ) from e


# -------------------------------
# Translation logic
# -------------------------------
def translate_to_english(text: str) -> str:
    tokenizer, model = load_translation_model()
    batch = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    translated = model.generate(**batch, max_length=512)
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return output[0] if output else ""


# -------------------------------
# Audit logging
# -------------------------------
def write_audit_log(record: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "input_text",
                "translated_text",
                "input_length",
                "status",
                "error_message",
            ]
        )

    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(rows)


# -------------------------------
# Session state
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------------
# UI
# -------------------------------
st.title("🌍 Translation Pipeline Demo")
st.caption(
    "A production-style translation app with caching, audit logging, history tracking, and monitoring metrics."
)

st.markdown(
    """
### What this app demonstrates
- cached model loading
- structured processing
- audit logging
- operational monitoring
- translation history tracking
- simple observability metrics
"""
)

col_left, col_right = st.columns([2, 1])

with col_left:
    textinp = st.text_area(
        "Enter non-English text",
        value="Cześć Proszę wprowadzić zdanie w języku innym niż angielski do przetłumaczenia",
        height=150,
    )

with col_right:
    st.subheader("Input Summary")
    st.metric("Characters", len(textinp))
    st.metric("Words", len(textinp.split()) if textinp.strip() else 0)


# -------------------------------
# Translate action
# -------------------------------
if st.button("Translate to English", use_container_width=True):
    event_time = datetime.utcnow().isoformat()

    if not textinp.strip():
        st.warning("Please enter some text before translating.")
    else:
        try:
            translated_text = translate_to_english(textinp)

            st.success("Translation completed successfully.")
            st.text_area("Translated English Text", translated_text, height=150)

            record = {
                "timestamp": event_time,
                "input_text": textinp,
                "translated_text": translated_text,
                "input_length": len(textinp),
                "status": "SUCCESS",
                "error_message": None,
            }

            st.session_state.history.append(record)
            write_audit_log(record)

        except ImportError as e:
            st.error(str(e))
            st.info("Install the missing packages in requirements.txt and redeploy the app.")

            record = {
                "timestamp": event_time,
                "input_text": textinp,
                "translated_text": None,
                "input_length": len(textinp),
                "status": "FAILED",
                "error_message": str(e),
            }
            st.session_state.history.append(record)
            write_audit_log(record)

        except Exception as e:
            error_msg = str(e)
            st.error("Translation failed.")
            st.code(error_msg)

            record = {
                "timestamp": event_time,
                "input_text": textinp,
                "translated_text": None,
                "input_length": len(textinp),
                "status": "FAILED",
                "error_message": error_msg,
            }
            st.session_state.history.append(record)
            write_audit_log(record)


# -------------------------------
# Monitoring / Analytics section
# -------------------------------
st.markdown("---")
st.subheader("📊 Translation Monitoring Dashboard")

audit_df = read_audit_log()

if audit_df.empty:
    st.info("No translation activity logged yet.")
else:
    total_requests = len(audit_df)
    success_count = int((audit_df["status"] == "SUCCESS").sum())
    failure_count = int((audit_df["status"] == "FAILED").sum())
    success_rate = round((success_count / total_requests) * 100, 2) if total_requests else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Requests", total_requests)
    m2.metric("Successful", success_count)
    m3.metric("Failed", failure_count)
    m4.metric("Success Rate", f"{success_rate}%")

    st.write("### Request Trend")
    render_request_trend(audit_df)

    st.write("### Audit Log")
    render_html_table(audit_df, height=320)


# -------------------------------
# Session History
# -------------------------------
st.markdown("---")
st.subheader("🧾 Current Session History")

session_df = pd.DataFrame(st.session_state.history)
if session_df.empty:
    st.info("No translations in this session yet.")
else:
    render_html_table(session_df, height=280)