import streamlit as st
import requests
from bs4 import BeautifulSoup
import subprocess
import json

# Function to call Ollama locally
def summarize_with_ollama(text, model="llama3"):
    try:
        # Run Ollama command with input prompt
        prompt = f"Summarize this news headline in one short sentence:\n\n{text}"
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

# Function to fetch Google News headlines (RSS)
def fetch_google_news():
    url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")[:10]  # Top 10 news
    news_list = []
    for item in items:
        title = item.title.text
        link = item.link.text
        news_list.append({"title": title, "link": link})
    return news_list

# Streamlit UI
st.title("🌍 Global News Topic Tracker (Ollama LLM)")
st.write("Scrape Google News and summarize trending topics using local Ollama models.")

if st.button("Fetch Latest News"):
    with st.spinner("Fetching latest news..."):
        news_items = fetch_google_news()

        for i, news in enumerate(news_items, 1):
            st.subheader(f"{i}. {news['title']}")
            st.markdown(f"[Read more]({news['link']})")

            with st.spinner("Summarizing with Ollama..."):
                summary = summarize_with_ollama(news['title'])
                st.write("**AI Summary:**", summary)

            st.markdown("---")
