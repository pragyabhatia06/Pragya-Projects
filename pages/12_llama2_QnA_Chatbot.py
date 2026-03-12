import json
import time
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="LLM Q&A Monitoring App",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------
# Config
# -----------------------------------
LOG_FILE = Path("chat_audit_log.jsonl")

DEFAULT_SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer clearly and concisely.
If the question is unclear, say what is missing.
Prefer practical examples related to data engineering, SQL, pipelines, cloud, ML systems, APIs, Spark, Airflow, and monitoring.
""".strip()

MODEL_OPTIONS = {
    "Phi-3 Mini": "microsoft/Phi-3-mini-4k-instruct",
    "TinyLlama Chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# -----------------------------------
# Helper functions
# -----------------------------------
def supports_chat_ui() -> bool:
    return hasattr(st, "chat_message") and hasattr(st, "chat_input")


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Ask me anything about data engineering, pipelines, SQL, cloud, or ML systems."
            }
        ]

    if "chat_metrics" not in st.session_state:
        st.session_state.chat_metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "last_latency_sec": None,
            "last_mode": "N/A",
        }


def clear_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything about data engineering, pipelines, SQL, cloud, or ML systems."
        }
    ]


def build_prompt(messages, system_prompt):
    prompt_parts = [f"System: {system_prompt}\n"]
    for msg in messages:
        role = msg["role"].capitalize()
        prompt_parts.append(f"{role}: {msg['content']}\n")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def log_interaction(user_prompt, assistant_response, model_name, latency, status, mode, error_message=None):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
        "mode": mode,
        "user_prompt": user_prompt,
        "assistant_response": assistant_response,
        "latency_sec": round(latency, 3) if latency is not None else None,
        "status": status,
        "error_message": error_message,
        "prompt_length": len(user_prompt) if user_prompt else 0,
        "response_length": len(assistant_response) if assistant_response else 0,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def query_huggingface(model_id: str, prompt: str, hf_token: str, max_new_tokens: int, temperature: float, top_p: float):
    """
    Uses Hugging Face Inference API.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True
        }
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()

    if isinstance(data, dict) and "error" in data:
        return f"Hugging Face API error: {data['error']}"

    return str(data)


def extract_recent_context(messages, max_items=4):
    """
    Take last few non-system chat items for fallback context.
    """
    filtered = [m for m in messages if m["role"] in {"user", "assistant"}]
    return filtered[-max_items:]


def smart_keyword_match(text, keywords):
    text_l = text.lower()
    return any(k in text_l for k in keywords)


def generate_fallback_response(user_prompt: str, messages=None, system_prompt: str = "") -> str:
    """
    API-key-free fallback responder.
    This is not a real LLM, but a portfolio-safe smart response generator.
    It gives useful structured answers when external inference is unavailable.
    """
    prompt = (user_prompt or "").strip()
    prompt_l = prompt.lower()

    recent_context = extract_recent_context(messages or [])
    context_hint = ""
    if len(recent_context) > 1:
        last_user_msgs = [m["content"] for m in recent_context if m["role"] == "user"]
        if last_user_msgs:
            context_hint = f"\n\nContext considered: {last_user_msgs[-2:] if len(last_user_msgs) >= 2 else last_user_msgs}"

    if not prompt:
        return "Please type a question and I’ll help."

    # Greetings
    if smart_keyword_match(prompt_l, ["hi", "hello", "hey", "good morning", "good evening"]):
        return (
            "Hello! I can help with:\n"
            "- SQL and PostgreSQL\n"
            "- Data engineering pipelines\n"
            "- Airflow, Spark, Databricks, dbt\n"
            "- APIs, monitoring, logging, Streamlit\n"
            "- ML system design\n\n"
            "Try asking something like: 'Explain medallion architecture with example' or 'Write a SQL query for deduplication.'"
        )

    # SQL
    if smart_keyword_match(prompt_l, ["sql", "postgres", "postgresql", "join", "cte", "window function", "query"]):
        return (
            "Here’s a practical SQL way to think about it:\n\n"
            "1. Start from the business question.\n"
            "2. Identify source tables and join keys.\n"
            "3. Filter early.\n"
            "4. Use CTEs for readability.\n"
            "5. Use window functions for dedup/ranking.\n"
            "6. Add indexes only for repeated access patterns.\n\n"
            "Example pattern:\n"
            "WITH base AS (\n"
            "    SELECT *\n"
            "    FROM orders\n"
            "    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'\n"
            "), ranked AS (\n"
            "    SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn\n"
            "    FROM base\n"
            ")\n"
            "SELECT *\n"
            "FROM ranked\n"
            "WHERE rn = 1;\n\n"
            "Send me your table structure or query and I’ll tailor it."
            f"{context_hint}"
        )

    # Data engineering / ETL
    if smart_keyword_match(prompt_l, ["etl", "elt", "pipeline", "data engineering", "ingestion", "medallion"]):
        return (
            "A clean data pipeline usually looks like this:\n\n"
            "1. Ingestion layer\n"
            "   - Pull from APIs, files, databases, or streams\n"
            "   - Store raw data with minimal transformation\n\n"
            "2. Bronze/raw layer\n"
            "   - Preserve source structure\n"
            "   - Add ingestion timestamp, source file name, batch id\n\n"
            "3. Silver/clean layer\n"
            "   - Standardize schema\n"
            "   - Deduplicate\n"
            "   - Validate nulls, formats, datatypes\n\n"
            "4. Gold/business layer\n"
            "   - KPIs, aggregates, dimension/fact models\n"
            "   - Ready for BI, ML, APIs\n\n"
            "5. Orchestration and monitoring\n"
            "   - Airflow/ADF/Prefect\n"
            "   - Log row counts, latency, failures, freshness\n\n"
            "6. Consumption\n"
            "   - Dashboards, APIs, notebooks, alerting\n\n"
            "If you want, ask me for a version using Airflow + PostgreSQL + Streamlit or Databricks + dbt."
            f"{context_hint}"
        )

    # Airflow
    if smart_keyword_match(prompt_l, ["airflow", "dag", "scheduler", "orchestration"]):
        return (
            "Airflow is best used for orchestration, not heavy transformation.\n\n"
            "Typical Airflow responsibilities:\n"
            "- schedule jobs\n"
            "- manage task dependencies\n"
            "- retry failed tasks\n"
            "- alert on failures\n"
            "- log execution metadata\n\n"
            "Typical DAG flow:\n"
            "extract_data >> validate_data >> transform_data >> load_data >> quality_checks >> notify\n\n"
            "Keep transformations in SQL/Spark/dbt scripts, and let Airflow trigger them."
            f"{context_hint}"
        )

    # Spark / Databricks
    if smart_keyword_match(prompt_l, ["spark", "databricks", "pyspark", "delta"]):
        return (
            "For Spark/Databricks, focus on distributed processing and optimization:\n\n"
            "- Use Spark for large-scale transformations\n"
            "- Partition by high-selectivity columns used in filtering\n"
            "- Avoid too many small files\n"
            "- Cache only reused intermediate DataFrames\n"
            "- Use Delta for ACID, schema evolution, and time travel\n"
            "- Use window functions carefully because they can be expensive\n\n"
            "Example dedup pattern:\n"
            "df_filtered = df.filter(col('id').isNotNull())\n"
            "window_spec = Window.partitionBy('business_key').orderBy(col('updated_at').desc())\n"
            "df_final = df_filtered.withColumn('rn', row_number().over(window_spec)).filter(col('rn') == 1).drop('rn')"
            f"{context_hint}"
        )

    # Monitoring / logging
    if smart_keyword_match(prompt_l, ["monitor", "monitoring", "logging", "observability", "metrics"]):
        return (
            "For a portfolio-ready monitoring setup, capture these fields for every interaction or job run:\n\n"
            "- timestamp\n"
            "- request id / run id\n"
            "- input prompt or source\n"
            "- output or status\n"
            "- latency\n"
            "- error message\n"
            "- model/job name\n"
            "- token or row counts if available\n\n"
            "Good dashboard KPIs:\n"
            "- total requests\n"
            "- total failures\n"
            "- average latency\n"
            "- success rate\n"
            "- most common errors\n"
            "- recent activity log\n\n"
            "Your app already does the right thing conceptually by storing JSONL audit records."
            f"{context_hint}"
        )

    # ML systems
    if smart_keyword_match(prompt_l, ["ml", "machine learning", "model", "feature engineering", "training"]):
        return (
            "A practical ML system has these layers:\n\n"
            "1. Data ingestion\n"
            "2. Feature engineering\n"
            "3. Training pipeline\n"
            "4. Validation and experiment tracking\n"
            "5. Model registry\n"
            "6. Inference serving\n"
            "7. Monitoring for drift, latency, and errors\n\n"
            "For interviews, explain not just the model, but also data freshness, feature consistency, retraining triggers, and deployment design."
            f"{context_hint}"
        )

    # API design
    if smart_keyword_match(prompt_l, ["api", "fastapi", "flask", "endpoint", "rest"]):
        return (
            "A good API design answer usually includes:\n\n"
            "- endpoint purpose\n"
            "- request schema\n"
            "- response schema\n"
            "- auth mechanism\n"
            "- validation\n"
            "- error handling\n"
            "- logging\n"
            "- pagination/filtering if needed\n\n"
            "Example:\n"
            "GET /news?category=data-engineering&limit=20\n\n"
            "Response:\n"
            "{\n"
            '  "status": "success",\n'
            '  "count": 20,\n'
            '  "data": [...]\n'
            "}\n\n"
            "If you want, I can generate a FastAPI example for your use case."
            f"{context_hint}"
        )

    # Default smart response
    return (
        "I can help with that. Here’s a practical way to approach it:\n\n"
        f"Question received: {prompt}\n\n"
        "1. Clarify the business or technical goal.\n"
        "2. Identify the input data, system, or API involved.\n"
        "3. Define the processing or logic needed.\n"
        "4. Decide the output format: SQL, Python, architecture, dashboard, or explanation.\n"
        "5. Add monitoring, validation, and error handling.\n\n"
        "Ask me again with one of these formats for a sharper answer:\n"
        "- 'Explain this concept'\n"
        "- 'Write SQL for this'\n"
        "- 'Give Python code'\n"
        "- 'Design architecture for this'\n"
        "- 'Prepare interview answer for this'"
        f"{context_hint}"
    )


def get_safe_hf_token():
    """
    Safer secret access across environments.
    """
    try:
        return st.secrets["HF_API_TOKEN"]
    except Exception:
        return ""


def show_messages():
    if supports_chat_ui():
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    else:
        st.warning("Your Streamlit version does not support chat components. Showing fallback layout.")
        for msg in st.session_state.messages:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")


# -----------------------------------
# Init
# -----------------------------------
init_session()

# -----------------------------------
# Sidebar
# -----------------------------------
with st.sidebar:
    st.title("🤖 LLM Q&A App")
    st.caption("Open-model chatbot with logging and monitoring")

    hf_token = get_safe_hf_token()
    manual_hf_token = ""

    if hf_token:
        st.success("Hugging Face API token found in secrets.")
    else:
        st.info("No Hugging Face token found. App will use built-in fallback response mode.")
        manual_hf_token = st.text_input("Optional: Enter Hugging Face API token", type="password")

    effective_hf_token = hf_token or manual_hf_token

    selected_model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
    selected_model_id = MODEL_OPTIONS[selected_model_label]

    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, 32)

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=140)

    st.button("Clear Chat History", on_click=clear_chat)

    st.markdown("---")
    st.subheader("Run Metrics")
    st.metric("Total Requests", st.session_state.chat_metrics["total_requests"])
    st.metric("Total Errors", st.session_state.chat_metrics["total_errors"])

    last_latency = st.session_state.chat_metrics["last_latency_sec"]
    st.metric("Last Latency (sec)", f"{last_latency:.2f}" if last_latency is not None else "N/A")

    st.metric("Last Mode", st.session_state.chat_metrics.get("last_mode", "N/A"))

# -----------------------------------
# Main UI
# -----------------------------------
st.title("LLM Q&A + Monitoring Dashboard")
st.write(
    "A portfolio-ready chatbot project demonstrating model integration, logging, monitoring, "
    "fallback handling, and robust Streamlit UI behavior."
)

show_messages()

# -----------------------------------
# Input
# -----------------------------------
if supports_chat_ui():
    user_prompt = st.chat_input("Ask a question...")
else:
    user_prompt = st.text_input("Ask a question...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    if supports_chat_ui():
        with st.chat_message("user"):
            st.write(user_prompt)
    else:
        st.markdown(f"**User:** {user_prompt}")

    assistant_response = ""
    start_time = time.time()
    mode_used = "fallback-no-token"

    try:
        full_prompt = build_prompt(st.session_state.messages, system_prompt)

        # Use Hugging Face if token exists, otherwise use built-in fallback response generator
        if effective_hf_token:
            mode_used = "huggingface-api"
            if supports_chat_ui():
                with st.chat_message("assistant"):
                    with st.spinner("Generating response from Hugging Face model..."):
                        assistant_response = query_huggingface(
                            model_id=selected_model_id,
                            prompt=full_prompt,
                            hf_token=effective_hf_token,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        st.write(assistant_response)
            else:
                with st.spinner("Generating response from Hugging Face model..."):
                    assistant_response = query_huggingface(
                        model_id=selected_model_id,
                        prompt=full_prompt,
                        hf_token=effective_hf_token,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                st.markdown(f"**Assistant:** {assistant_response}")
        else:
            mode_used = "smart-fallback"
            if supports_chat_ui():
                with st.chat_message("assistant"):
                    with st.spinner("Generating response using built-in fallback engine..."):
                        assistant_response = generate_fallback_response(
                            user_prompt=user_prompt,
                            messages=st.session_state.messages,
                            system_prompt=system_prompt
                        )
                        st.write(assistant_response)
            else:
                with st.spinner("Generating response using built-in fallback engine..."):
                    assistant_response = generate_fallback_response(
                        user_prompt=user_prompt,
                        messages=st.session_state.messages,
                        system_prompt=system_prompt
                    )
                st.markdown(f"**Assistant:** {assistant_response}")

        latency = time.time() - start_time
        st.session_state.chat_metrics["total_requests"] += 1
        st.session_state.chat_metrics["last_latency_sec"] = latency
        st.session_state.chat_metrics["last_mode"] = mode_used

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        log_interaction(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            model_name=selected_model_id if effective_hf_token else "built-in-fallback",
            latency=latency,
            status="success",
            mode=mode_used
        )

    except Exception as e:
        latency = time.time() - start_time
        st.session_state.chat_metrics["total_errors"] += 1
        st.session_state.chat_metrics["last_latency_sec"] = latency
        st.session_state.chat_metrics["last_mode"] = "error"

        error_text = f"Error: {str(e)}"
        st.error(error_text)

        # Final safety fallback so app still answers even if HF call fails unexpectedly
        try:
            assistant_response = generate_fallback_response(
                user_prompt=user_prompt,
                messages=st.session_state.messages,
                system_prompt=system_prompt
            )

            if supports_chat_ui():
                with st.chat_message("assistant"):
                    st.write(assistant_response)
            else:
                st.markdown(f"**Assistant:** {assistant_response}")

            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.info("Switched to built-in fallback mode after external model failure.")

            log_interaction(
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                model_name="built-in-fallback",
                latency=latency,
                status="success-with-fallback-after-error",
                mode="fallback-after-error",
                error_message=str(e)
            )
        except Exception as inner_e:
            log_interaction(
                user_prompt=user_prompt,
                assistant_response="",
                model_name=selected_model_id,
                latency=latency,
                status="failed",
                mode="error",
                error_message=f"{str(e)} | fallback_error={str(inner_e)}"
            )

# -----------------------------------
# Optional log preview
# -----------------------------------
st.markdown("---")
st.subheader("Recent Chat Audit Log")

if LOG_FILE.exists():
    try:
        rows = []
        with LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f.readlines()[-10:]:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if rows:
            df_logs = pd.DataFrame(rows).fillna("")

            # Avoid Streamlit / Arrow dtype issues
            for col in df_logs.columns:
                df_logs[col] = df_logs[col].astype(str)

            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No logs yet. Start chatting to generate monitoring records.")
    except Exception as e:
        st.warning(f"Could not read log file: {e}")
        # extra-safe raw display
        try:
            with LOG_FILE.open("r", encoding="utf-8") as f:
                raw_lines = f.readlines()[-10:]
            st.text("".join(raw_lines))
        except Exception:
            pass
else:
    st.info("No logs yet. Start chatting to generate monitoring records.")