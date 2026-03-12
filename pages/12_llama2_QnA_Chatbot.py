import os
import time
import json
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
            {"role": "assistant", "content": "Hello! Ask me anything about data engineering, pipelines, SQL, cloud, or ML systems."}
        ]
    if "chat_metrics" not in st.session_state:
        st.session_state.chat_metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "last_latency_sec": None,
        }

def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about data engineering, pipelines, SQL, cloud, or ML systems."}
    ]

def build_prompt(messages, system_prompt):
    prompt_parts = [f"System: {system_prompt}\n"]
    for msg in messages:
        role = msg["role"].capitalize()
        prompt_parts.append(f"{role}: {msg['content']}\n")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

def log_interaction(user_prompt, assistant_response, model_name, latency, status, error_message=None):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model_name,
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

    return str(data)

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

    hf_token = st.secrets.get("HF_API_TOKEN", "")
    if hf_token:
        st.success("Hugging Face API token found in secrets.")
    else:
        hf_token = st.text_input("Enter Hugging Face API token", type="password")

    selected_model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
    selected_model_id = MODEL_OPTIONS[selected_model_label]

    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, 32)

    system_prompt = st.text_area("System prompt", value=DEFAULT_SYSTEM_PROMPT, height=120)

    st.button("Clear Chat History", on_click=clear_chat)

    st.markdown("---")
    st.subheader("Run Metrics")
    st.metric("Total Requests", st.session_state.chat_metrics["total_requests"])
    st.metric("Total Errors", st.session_state.chat_metrics["total_errors"])
    last_latency = st.session_state.chat_metrics["last_latency_sec"]
    st.metric("Last Latency (sec)", f"{last_latency:.2f}" if last_latency is not None else "N/A")

# -----------------------------------
# Main UI
# -----------------------------------
st.title("LLM Q&A + Monitoring Dashboard")
st.write("A portfolio-ready chatbot project demonstrating model integration, logging, monitoring, and robust Streamlit UI handling.")

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

    try:
        if not hf_token:
            raise ValueError("Missing Hugging Face API token.")

        full_prompt = build_prompt(st.session_state.messages, system_prompt)

        if supports_chat_ui():
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    assistant_response = query_huggingface(
                        model_id=selected_model_id,
                        prompt=full_prompt,
                        hf_token=hf_token,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    st.write(assistant_response)
        else:
            with st.spinner("Generating response..."):
                assistant_response = query_huggingface(
                    model_id=selected_model_id,
                    prompt=full_prompt,
                    hf_token=hf_token,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            st.markdown(f"**Assistant:** {assistant_response}")

        latency = time.time() - start_time
        st.session_state.chat_metrics["total_requests"] += 1
        st.session_state.chat_metrics["last_latency_sec"] = latency

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        log_interaction(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            model_name=selected_model_id,
            latency=latency,
            status="success"
        )

    except Exception as e:
        latency = time.time() - start_time
        st.session_state.chat_metrics["total_errors"] += 1
        st.session_state.chat_metrics["last_latency_sec"] = latency

        error_text = f"Error: {str(e)}"
        st.error(error_text)

        log_interaction(
            user_prompt=user_prompt,
            assistant_response="",
            model_name=selected_model_id,
            latency=latency,
            status="failed",
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
                rows.append(json.loads(line))
        if rows:
            st.dataframe(rows, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not read log file: {e}")
else:
    st.info("No logs yet. Start chatting to generate monitoring records.")