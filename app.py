import json
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from chat_manager import ChatHistoryManager

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
history_mgr = ChatHistoryManager("chat_history.db")

st.set_page_config(
    page_title="AI Chat Bot",
    page_icon="",
    layout="wide",
)

GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-2.5-pro",
]


@st.cache_data(ttl=600)
def get_available_gemini_models() -> list[str]:
    if not gemini_client:
        return []
    found = []
    try:
        for m in gemini_client.models.list():
            name = (m.name or "").replace("models/", "")
            if not name:
                continue
            if any(skip in name for skip in ("embedding", "image", "tts", "veo", "imagen", "deep-research")):
                continue
            if name.startswith(("gemini-", "gemma-")):
                found.append(name)
    except Exception:
        return GEMINI_FALLBACK_MODELS

    if not found:
        return GEMINI_FALLBACK_MODELS

    ordered = [m for m in GEMINI_FALLBACK_MODELS if m in found]
    for m in found:
        if m not in ordered:
            ordered.append(m)
    return ordered


def get_available_ollama_models() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
            if models:
                return models
    except Exception:
        pass
    return []


AVAILABLE_GEMINI_MODELS = get_available_gemini_models()
AVAILABLE_OLLAMA_MODELS = get_available_ollama_models()

if not AVAILABLE_GEMINI_MODELS and not AVAILABLE_OLLAMA_MODELS:
    st.error("GEMINI_API_KEY .env file eke add karanna naththam Ollama run karanna.")
    st.stop()

if "all_chats" not in st.session_state:
    st.session_state.all_chats = history_mgr.load_all()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = history_mgr.generate_chat_id()

if "messages" not in st.session_state:
    st.session_state.messages = []


def persist_current_chat() -> None:
    st.session_state.all_chats = history_mgr.upsert_chat(
        st.session_state.all_chats,
        st.session_state.current_chat_id,
        st.session_state.messages,
    )


engine_options = []
if AVAILABLE_OLLAMA_MODELS:
    engine_options.append("Local Ollama")
if AVAILABLE_GEMINI_MODELS:
    engine_options.append("Gemini Cloud")


# --- Sidebar: Mode, New Chat & History ---
with st.sidebar:
    st.header("Assistant Settings")

    mode = st.radio(
        "Response Mode",
        [
            "General Assistant",
            "Coding & Tech Pro",
            "Quick Concise",
            "Brainstorm & Ideas",
        ],
    )

    if st.button("➕ New Chat", use_container_width=True):
        persist_current_chat()
        st.session_state.current_chat_id = history_mgr.generate_chat_id()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("💬 Chat History")

    sorted_chat_ids = sorted(st.session_state.all_chats.keys(), reverse=True)
    if not sorted_chat_ids:
        st.caption("Not saved chat.")
    else:
        for cid in sorted_chat_ids:
            chat_item = st.session_state.all_chats[cid]
            title = chat_item.get("title", "Untitled Chat")
            is_active = cid == st.session_state.current_chat_id

            col1, col2 = st.columns([5, 1])
            with col1:
                label = f"▶ {title}" if is_active else title
                if st.button(label, key=f"load_{cid}", use_container_width=True):
                    persist_current_chat()
                    st.session_state.current_chat_id = cid
                    st.session_state.messages = list(chat_item.get("messages", []))
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{cid}", help="Delete chat"):
                    st.session_state.all_chats = history_mgr.delete_chat(
                        st.session_state.all_chats, cid
                    )
                    if cid == st.session_state.current_chat_id:
                        st.session_state.current_chat_id = history_mgr.generate_chat_id()
                        st.session_state.messages = []
                    st.rerun()


st.title("🤖 AI Assistant")


def build_system_instruction(selected_mode: str) -> str:
    return f"""
You are a helpful, smart, and reliable general-purpose AI assistant with strong capabilities in programming, software development, and technical problem-solving.
Current Mode: {selected_mode}

Core guidelines:
1. Match the user's language naturally (English, Sinhala, or Singlish) while keeping technical terms clear and accurate.
2. Act as an all-round assistant for everyday questions, explanations, planning, writing, and general tasks.
3. Give extra care and depth to programming, debugging, system design, and technical workflows: provide clean, working code, explain key logic briefly, and highlight important caveats or edge cases when helpful.
4. In "General Assistant" mode: give balanced, clear, and practical answers.
5. In "Coding & Tech Pro" mode: prioritize precision, clean code structure, edge cases, and actionable technical steps.
6. In "Quick Concise" mode: keep answers short, direct, and easy to scan.
7. In "Brainstorm & Ideas" mode: offer structured options, creative angles, and practical next steps.
"""


def build_gemini_contents(history: list[dict]) -> list[types.Content]:
    return [
        types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])],
        )
        for msg in history
    ]


def stream_ollama(history: list[dict], sys_prompt: str, model_name: str):
    messages = [{"role": "system", "content": sys_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model_name, "messages": messages, "stream": True},
        stream=True,
        timeout=90,
    )
    resp.raise_for_status()

    yielded = False
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        content = data.get("message", {}).get("content")
        if content:
            yielded = True
            yield content
    return yielded


def stream_gemini_with_fallback(history: list[dict], preferred_model: str, sys_prompt: str):
    if not AVAILABLE_GEMINI_MODELS:
        return None
    models_queue = [preferred_model] + [m for m in AVAILABLE_GEMINI_MODELS if m != preferred_model]
    contents = build_gemini_contents(history)

    for model_name in models_queue:
        try:
            if model_name.startswith("gemma"):
                merged_contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"Instructions:\n{sys_prompt}")],
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Understood. I will follow these instructions.")],
                    ),
                ] + contents
                stream = gemini_client.models.generate_content_stream(
                    model=model_name,
                    contents=merged_contents,
                    config=types.GenerateContentConfig(temperature=0.6),
                )
            else:
                stream = gemini_client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_prompt,
                        temperature=0.6,
                    ),
                )

            chunks = []
            for chunk in stream:
                if chunk.text:
                    chunks.append(chunk.text)
                    yield chunk.text

            if chunks:
                return model_name
        except Exception:
            continue
    return None


def stream_unified_response(history: list[dict], engine: str, selected_mode: str):
    sys_prompt = build_system_instruction(selected_mode)
    preferred_ollama = AVAILABLE_OLLAMA_MODELS[0] if AVAILABLE_OLLAMA_MODELS else ""
    preferred_gemini = AVAILABLE_GEMINI_MODELS[0] if AVAILABLE_GEMINI_MODELS else ""

    if engine.startswith("Local Ollama") and preferred_ollama:
        try:
            ok = yield from stream_ollama(history, sys_prompt, preferred_ollama)
            if ok:
                return
        except Exception as e:
            if preferred_gemini:
                used = yield from stream_gemini_with_fallback(history, preferred_gemini, sys_prompt)
                if used:
                    yield f"\n\n*(Switched to Gemini `{used}`)*"
                    return
            yield f"Ollama error: `{e}`"
            return

    if preferred_gemini:
        used_gemini = yield from stream_gemini_with_fallback(history, preferred_gemini, sys_prompt)
        if used_gemini:
            if used_gemini != preferred_gemini:
                yield f"\n\n*(Answered using `{used_gemini}`)*"
            return

    if preferred_ollama:
        try:
            ok = yield from stream_ollama(history, sys_prompt, preferred_ollama)
            if ok:
                yield f"\n\n*(Gemini busy — answered using Local Ollama `{preferred_ollama}`)*"
                return
        except Exception:
            pass

    yield "Configured models okkoma danata busy ho unavailable. Ollama run wenawada balanna naththam tika welawakin try karanna."


# --- Render Chat Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown("<div style='height: 135px;'></div>", unsafe_allow_html=True)

# --- Right-Aligned Engine Pill Above Chat Input (No Model Dropdown) ---
st.markdown(
    """
    <style>
    .block-container {
        padding-bottom: 9rem !important;
    }
    .st-key-right_engine_pill {
        position: fixed;
        bottom: 82px;
        right: 3.5rem;
        width: 175px;
        z-index: 998;
    }
    .st-key-right_engine_pill [data-testid="stSelectbox"] > div > div {
        background-color: #26282c !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        min-height: 30px !important;
        height: 30px !important;
        font-size: 0.78rem !important;
    }
    .st-key-right_engine_pill [data-testid="stSelectbox"] {
        margin-bottom: 0 !important;
    }
    div[data-testid="stBottom"] {
        background: transparent !important;
        padding-bottom: 0.85rem !important;
    }
    div[data-testid="stBottom"] > div {
        max-width: 780px;
        margin: 0 auto;
    }
    div[data-testid="stChatInput"] > div {
        background-color: #1e1f20 !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 28px !important;
    }
    @media (max-width: 900px) {
        .st-key-right_engine_pill {
            right: 1.5rem;
            width: 150px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="right_engine_pill"):
    selected_engine = st.selectbox(
        "AI Engine",
        engine_options,
        index=0,
        key="final_right_engine_select",
        label_visibility="collapsed",
    )

if prompt := st.chat_input("Ask anything or paste code/errors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    persist_current_chat()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply_text = st.write_stream(
            stream_unified_response(st.session_state.messages, selected_engine, mode)
        )

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    persist_current_chat()
    st.rerun()