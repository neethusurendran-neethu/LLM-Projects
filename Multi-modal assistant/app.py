import streamlit as st
import requests
import base64
import tempfile
import json
import whisper

# ---------------------------
# CONFIG
# ---------------------------
OLLAMA_API = "http://localhost:11434/api/generate"

st.set_page_config(page_title="🦙 LLM Multimedia Assistant", layout="wide")
st.title("🦙 LLM Multimedia Assistant (Streamlit + Ollama)")

# ---------------------------
# HELPER: handle streaming responses from Ollama
# ---------------------------
def ollama_generate(payload):
    """Send prompt to Ollama API and collect streaming response."""
    response = requests.post(OLLAMA_API, json=payload, stream=True)
    full_reply = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    full_reply += data["response"]
            except json.JSONDecodeError:
                continue
    return full_reply.strip()

# ---------------------------
# TEXT CHAT
# ---------------------------
st.header("💬 Chat with Ollama (Text Models)")
user_input = st.text_input("Enter your message:")

if st.button("Send Text"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            reply = ollama_generate({
                "model": "llama3",
                "prompt": user_input
            })
            st.success(reply)

# ---------------------------
# IMAGE ANALYSIS
# ---------------------------
st.header("🖼️ Image Understanding with Ollama (llava)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        img_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        with st.spinner("Analyzing image..."):
            desc = ollama_generate({
                "model": "llava",
                "prompt": "Describe this image in detail",
                "images": [img_b64],
            })
            st.success(desc)

# ---------------------------
# AUDIO TRANSCRIPTION
# ---------------------------
st.header("🎙️ Audio Transcription with Whisper")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
if audio_file is not None:
    st.audio(audio_file)

    if st.button("Transcribe Audio"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        with st.spinner("Transcribing audio..."):
            model = whisper.load_model("base")  # you can use "small" or "medium" if you want more accuracy
            result = model.transcribe(tmp_path)
            st.success(result["text"])
