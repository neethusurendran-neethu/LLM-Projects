# app.py (frontend with Streamlit)
import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/summarize/"

st.title("📄 PDF Summarizer")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Summarizing..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(BACKEND_URL, files=files)
        if response.status_code == 200:
            st.subheader("Summary:")
            st.write(response.json()["summary"])
        else:
            st.error("Something went wrong")
