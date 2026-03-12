import json
import time
from datetime import datetime
from pathlib import Path

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
# Helpers
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


def get_safe_hf_token():
    try:
        return st.secrets["HF_API_TOKEN"]
    except Exception:
        return ""


def query_huggingface(model_id: str, prompt: str, hf_token: str, max_new_tokens: int, temperature: float, top_p: float):
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False,
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
        raise ValueError(data["error"])

    return str(data)


def generate_fallback_response(user_prompt: str) -> str:
    q = (user_prompt or "").strip().lower()

    if not q:
        return "Please type a question and I’ll help."

    if q in ["hi", "hello", "hey"] or q.startswith("hi ") or q.startswith("hello "):
        return (
            "Hello! I can help with SQL, PostgreSQL, pipelines, Airflow, Spark, APIs, and Streamlit projects."
        )

    if "sql" in q or "postgres" in q or "postgresql" in q:
        return (
            "SQL is used to store, retrieve, filter, join, and analyze structured data in databases.\n\n"
            "Common operations:\n"
            "- SELECT: read data\n"
            "- WHERE: filter rows\n"
            "- JOIN: combine tables\n"
            "- GROUP BY: aggregate data\n"
            "- WINDOW FUNCTIONS: ranking, deduplication, running totals\n\n"
            "Example:\n"
            "SELECT customer_id, SUM(amount) AS total_amount\n"
            "FROM orders\n"
            "WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'\n"
            "GROUP BY customer_id;\n\n"
            "If you want, ask a more specific SQL question and I’ll explain with an example."
        )

    if "airflow" in q or "dag" in q:
        return (
            "Airflow is mainly used for orchestration.\n\n"
            "Typical responsibilities:\n"
            "- schedule jobs\n"
            "- manage dependencies\n"
            "- retry failures\n"
            "- send alerts\n"
            "- track task execution\n\n"
            "Typical flow:\n"
            "extract >> validate >> transform >> load >> quality_check >> notify"
        )

    if "spark" in q or "databricks" in q or "pyspark" in q:
        return (
            "Spark is used for distributed data processing on large datasets.\n\n"
            "Common optimization ideas:\n"
            "- filter early\n"
            "- avoid too many small files\n"
            "- partition carefully\n"
            "- cache only reused DataFrames\n"
            "- use window functions carefully"
        )

    if "api" in q or "fastapi" in q or "flask" in q:
        return (
            "A good API design should include:\n"
            "- endpoint purpose\n"
            "- request/response schema\n"
            "- authentication\n"
            "- validation\n"
            "- error handling\n"
            "- logging and monitoring"
        )

    if "pipeline" in q or "etl" in q or "elt" in q:
        return (
            "A practical pipeline usually has:\n"
            "1. ingestion\n"
            "2. raw/bronze layer\n"
            "3. clean/silver layer\n"
            "4. business/gold layer\n"
            "5. orchestration\n"
            "6. monitoring and alerts"
        )

    return (
        f"You asked: {user_prompt}\n\n"
        "No external API token is configured, so this answer is coming from the built-in fallback mode.\n"
        "Ask me about SQL, PostgreSQL, Airflow, Spark, ETL, APIs, or Streamlit."
    )


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

    secret_hf_token = get_safe_hf_token()

    if secret_hf_token:
        st.success("Hugging Face API token found in secrets.")
    else:
        st.info("No Hugging Face token found. App will use built-in fallback mode.")

    manual_hf_token = st.text_input("Optional: Enter Hugging Face API token", type="password")
    effective_hf_token = manual_hf_token.strip() or secret_hf_token

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
    st.metric("Last Mode", st.session_state.chat_metrics["last_mode"])

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
    mode_used = "smart-fallback"

    try:
        if effective_hf_token:
            mode_used = "huggingface-api"
            full_prompt = build_prompt(st.session_state.messages, system_prompt)

            assistant_response = query_huggingface(
                model_id=selected_model_id,
                prompt=full_prompt,
                hf_token=effective_hf_token,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        else:
            mode_used = "smart-fallback"
            assistant_response = generate_fallback_response(user_prompt)

        latency = time.time() - start_time

        if supports_chat_ui():
            with st.chat_message("assistant"):
                st.write(assistant_response)
        else:
            st.markdown(f"**Assistant:** {assistant_response}")

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.chat_metrics["total_requests"] += 1
        st.session_state.chat_metrics["last_latency_sec"] = latency
        st.session_state.chat_metrics["last_mode"] = mode_used

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
        st.session_state.chat_metrics["last_mode"] = "fallback-after-error"

        fallback_response = generate_fallback_response(user_prompt)

        st.error(f"Error: {str(e)}")

        if supports_chat_ui():
            with st.chat_message("assistant"):
                st.write(fallback_response)
        else:
            st.markdown(f"**Assistant:** {fallback_response}")

        st.session_state.messages.append({"role": "assistant", "content": fallback_response})

        log_interaction(
            user_prompt=user_prompt,
            assistant_response=fallback_response,
            model_name="built-in-fallback",
            latency=latency,
            status="success-with-fallback",
            mode="fallback-after-error",
            error_message=str(e)
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
            for i, row in enumerate(reversed(rows), 1):
                with st.expander(f"Log Entry {i} | {row.get('timestamp', 'N/A')}"):
                    st.json(row)
        else:
            st.info("No logs yet. Start chatting to generate monitoring records.")
    except Exception as e:
        st.warning(f"Could not read log file: {e}")
else:
    st.info("No logs yet. Start chatting to generate monitoring records.")