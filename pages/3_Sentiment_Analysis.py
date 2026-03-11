from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Text Sentiment Pipeline Dashboard",
    page_icon="📈",
    layout="wide",
)

LOG_FILE = Path("sentiment_audit_log.jsonl")


# ---------------------------
# Resource loading
# ---------------------------
@st.cache_resource
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_transformer_resources():
    """
    Optional transformer loader.
    Only used if user selects transformer mode.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model, softmax


# ---------------------------
# Core analysis functions
# ---------------------------
def analyze_vader(text: str) -> dict:
    sia = load_vader()
    scores = sia.polarity_scores(text)

    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "engine": "NLTK Vader",
        "label": label,
        "negative_score": scores["neg"],
        "neutral_score": scores["neu"],
        "positive_score": scores["pos"],
        "compound_score": scores["compound"],
    }


def analyze_roberta(text: str) -> dict:
    tokenizer, model, softmax = load_transformer_resources()
    encoded_text = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()
    probs = softmax(scores)

    labels = ["Negative", "Neutral", "Positive"]
    best_idx = probs.argmax()

    return {
        "engine": "RoBERTa",
        "label": labels[best_idx],
        "negative_score": float(probs[0]),
        "neutral_score": float(probs[1]),
        "positive_score": float(probs[2]),
        "compound_score": None,
    }


# ---------------------------
# Audit log helpers
# ---------------------------
def write_audit_log(record: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_audit_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "engine",
                "input_text",
                "text_length",
                "label",
                "negative_score",
                "neutral_score",
                "positive_score",
                "compound_score",
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


# ---------------------------
# Batch processing
# ---------------------------
def process_batch(df: pd.DataFrame, text_column: str, engine: str) -> pd.DataFrame:
    results = []

    for text in df[text_column].fillna("").astype(str):
        event_time = datetime.utcnow().isoformat()

        try:
            if engine == "NLTK Vader":
                result = analyze_vader(text)
            else:
                result = analyze_roberta(text)

            record = {
                "timestamp": event_time,
                "engine": result["engine"],
                "input_text": text,
                "text_length": len(text),
                "label": result["label"],
                "negative_score": result["negative_score"],
                "neutral_score": result["neutral_score"],
                "positive_score": result["positive_score"],
                "compound_score": result["compound_score"],
                "status": "SUCCESS",
                "error_message": None,
            }
        except Exception as e:
            record = {
                "timestamp": event_time,
                "engine": engine,
                "input_text": text,
                "text_length": len(text),
                "label": None,
                "negative_score": None,
                "neutral_score": None,
                "positive_score": None,
                "compound_score": None,
                "status": "FAILED",
                "error_message": str(e),
            }

        write_audit_log(record)
        results.append(record)

    return pd.DataFrame(results)


# ---------------------------
# UI
# ---------------------------
st.title("📈 Text Sentiment Pipeline Dashboard")
st.caption("A production-style sentiment app with batch processing, audit logs, and monitoring metrics.")

engine = st.radio(
    "Select analysis engine",
    ["NLTK Vader", "RoBERTa"],
    horizontal=True,
)

tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch CSV Upload", "Monitoring Dashboard"])

# ---------------------------
# Tab 1: single text
# ---------------------------
with tab1:
    textinp = st.text_area(
        "Enter text to analyze",
        value="I am a good developer and I enjoy solving real-world data problems.",
        height=140,
    )

    c1, c2 = st.columns([1, 4])
    with c1:
        run_single = st.button("Run Analysis", use_container_width=True)

    with c2:
        st.metric("Characters", len(textinp))
        st.metric("Words", len(textinp.split()) if textinp.strip() else 0)

    if run_single:
        event_time = datetime.utcnow().isoformat()

        try:
            if engine == "NLTK Vader":
                result = analyze_vader(textinp)
            else:
                result = analyze_roberta(textinp)

            st.subheader("Prediction Result")
            if result["label"] == "Positive":
                st.success(f"Label: {result['label']}")
            elif result["label"] == "Negative":
                st.error(f"Label: {result['label']}")
            else:
                st.info(f"Label: {result['label']}")

            result_df = pd.DataFrame([result])
            st.dataframe(result_df, use_container_width=True)

            record = {
                "timestamp": event_time,
                "engine": result["engine"],
                "input_text": textinp,
                "text_length": len(textinp),
                "label": result["label"],
                "negative_score": result["negative_score"],
                "neutral_score": result["neutral_score"],
                "positive_score": result["positive_score"],
                "compound_score": result["compound_score"],
                "status": "SUCCESS",
                "error_message": None,
            }
            write_audit_log(record)

        except Exception as e:
            st.error(f"Analysis failed: {e}")

            record = {
                "timestamp": event_time,
                "engine": engine,
                "input_text": textinp,
                "text_length": len(textinp),
                "label": None,
                "negative_score": None,
                "neutral_score": None,
                "positive_score": None,
                "compound_score": None,
                "status": "FAILED",
                "error_message": str(e),
            }
            write_audit_log(record)

# ---------------------------
# Tab 2: batch upload
# ---------------------------
with tab2:
    st.write("Upload a CSV with a text column for batch sentiment processing.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        st.dataframe(upload_df.head(10), use_container_width=True)

        if not upload_df.empty:
            text_column = st.selectbox("Select text column", upload_df.columns.tolist())

            if st.button("Run Batch Processing"):
                batch_result_df = process_batch(upload_df, text_column, engine)
                st.success("Batch processing completed.")
                st.dataframe(batch_result_df, use_container_width=True)

                csv_data = batch_result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results CSV",
                    data=csv_data,
                    file_name="sentiment_batch_results.csv",
                    mime="text/csv",
                )

# ---------------------------
# Tab 3: monitoring
# ---------------------------
with tab3:
    st.subheader("Operational Monitoring")

    audit_df = read_audit_log()

    if audit_df.empty:
        st.info("No sentiment requests logged yet.")
    else:
        total_requests = len(audit_df)
        success_count = (audit_df["status"] == "SUCCESS").sum()
        failure_count = (audit_df["status"] == "FAILED").sum()
        success_rate = round((success_count / total_requests) * 100, 2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Requests", total_requests)
        m2.metric("Successful", success_count)
        m3.metric("Failed", failure_count)
        m4.metric("Success Rate", f"{success_rate}%")

        audit_df["timestamp"] = pd.to_datetime(audit_df["timestamp"], errors="coerce")

        if audit_df["timestamp"].notna().any():
            trend_df = (
                audit_df.assign(request_date=audit_df["timestamp"].dt.date)
                .groupby(["request_date", "status"])
                .size()
                .reset_index(name="count")
                .pivot(index="request_date", columns="status", values="count")
                .fillna(0)
            )
            st.write("### Request Trend")
            st.line_chart(trend_df)

        st.write("### Label Distribution")
        label_df = audit_df[audit_df["status"] == "SUCCESS"]["label"].value_counts()
        st.bar_chart(label_df)

        st.write("### Audit Log")
        st.dataframe(audit_df, use_container_width=True)