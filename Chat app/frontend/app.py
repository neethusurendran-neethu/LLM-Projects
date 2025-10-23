import streamlit as st
import requests

st.set_page_config(page_title="💬 Local LLM Chat App")

# Add a sidebar for model selection
with st.sidebar:
    st.title("Settings")
    model_name = st.text_input("Enter Ollama model name:", "tinyllama:latest")

# Store chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Create two tabs
tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])

# --- Tab 1: Chat Interface ---
with tab1:
    st.title("💬 LLM Chat App")

    with st.form("chat_form"):
        user_input = st.text_input("Ask something:", key="user_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip() != "":
        # Add user message to history
        st.session_state.history.append(("You", user_input))

        # Send to backend
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"message": user_input, "model": model_name},
            )
            bot_reply = response.json().get("reply", "⚠️ No response")
        except Exception as e:
            bot_reply = f"❌ Error: {e}"

        # Add bot response to history
        st.session_state.history.append(("Bot", bot_reply))

        # Reload the UI to show updates
        st.rerun()

    # Show conversation
    for speaker, message in st.session_state.history:
        st.markdown(f"**{speaker}:** {message}")

# --- Tab 2: Full History ---
with tab2:
    st.title("📜 Chat History")

    if st.session_state.history:
        for speaker, message in st.session_state.history:
            st.markdown(f"**{speaker}:** {message}")
    else:
        st.write("No conversation yet.")
