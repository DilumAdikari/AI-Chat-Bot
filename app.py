import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY .env file eke hoyaganna ba. Karunakarala add karanna.")
    st.stop()

client = genai.Client(api_key=api_key)

st.set_page_config(
    page_title="General AI Assistant",
    page_icon="🤖",
    layout="wide",
)


@st.cache_data(ttl=600)
def get_available_models() -> list[str]:
    preferred_order = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.8-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemma-3-27b-it",
        "gemma-3-12b-it",
    ]
    available = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods and ("gemini" in name or "gemma" in name):
                if "embedding" not in name and "image" not in name and "tts" not in name:
                    available.append(name)
    except Exception:
        return preferred_order

    ordered = [m for m in preferred_order if m in available]
    for m in available:
        if m not in ordered:
            ordered.append(m)
    return ordered or preferred_order


AVAILABLE_MODELS = get_available_models()

with st.sidebar:
    st.header("⚙️ Assistant Settings")
    selected_model = st.selectbox(
        "Model (Auto-Fallback Enabled)",
        AVAILABLE_MODELS,
        index=0,
    )
    mode = st.radio(
        "Response Mode",
        [
            "General Assistant",
            "Coding & Tech Pro",
            "Quick Concise",
            "Brainstorm & Ideas",
        ],
    )
    if st.button("🔄 New Chat"):
        st.session_state.pop("messages", None)
        st.rerun()

st.title("🤖 AI Assistant")
st.caption(f"Mode: **{mode}** • Preferred Model: `{selected_model}`")


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


def build_contents(history: list[dict]) -> list[types.Content]:
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    return contents


def stream_with_instant_fallback(history: list[dict], preferred_model: str, selected_mode: str):
    models_queue = [preferred_model] + [m for m in AVAILABLE_MODELS if m != preferred_model]
    contents = build_contents(history)
    sys_prompt = build_system_instruction(selected_mode)

    last_error = ""
    for model_name in models_queue[:6]:
        try:
            # Gemma models samahara welawata system_instruction support nokarana nisa eka handle karanawa
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
                config = types.GenerateContentConfig(temperature=0.6)
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=merged_contents,
                    config=config,
                )
            else:
                config = types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=0.6,
                )
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=config,
                )

            got_chunk = False
            for chunk in stream:
                if chunk.text:
                    got_chunk = True
                    yield chunk.text

            if got_chunk:
                if model_name != preferred_model:
                    yield f"\n\n*(Answered using `{model_name}`)*"
                return
        except Exception as e:
            last_error = str(e)
            continue

    yield f"Models okkoma danata busy ho unavailable. Last error: `{last_error}`"


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything or paste code/errors..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply_text = st.write_stream(
            stream_with_instant_fallback(
                st.session_state.messages, selected_model, mode
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": reply_text})