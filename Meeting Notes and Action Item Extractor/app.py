import streamlit as st
import warnings
# Suppress all torch classes warnings
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from transformers import pipeline

# Load local summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("📋 Meeting Notes & Action Item Extractor (Offline)")

# Text input box
meeting_text = st.text_area("Paste your meeting transcript here:")

if st.button("Extract"):
    if meeting_text.strip():
        # Summarize text
        summary = summarizer(meeting_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']

        # Extract action items (rule-based)
        action_items = []
        for line in meeting_text.split("\n"):
            if "will" in line or "by" in line:
                action_items.append(line.strip())

        st.subheader("📌 Meeting Summary")
        st.write(summary)

        st.subheader("✅ Action Items")
        for item in action_items:
            st.write("- " + item)
    else:
        st.warning("⚠️ Please enter some text before extracting.")
