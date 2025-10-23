# Global News Topic Tracker

A Streamlit application that fetches trending news from Google News RSS feed and uses local Ollama models to summarize headlines using AI.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Architecture](#architecture)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- Fetches the latest news headlines from Google News RSS feed
- Summarizes news headlines using local Ollama AI models
- Built with Streamlit for an interactive web interface
- Uses BeautifulSoup for RSS parsing
- Leverages subprocess to interact with Ollama locally

## Prerequisites

- Python 3.7+
- Ollama running locally
- At least one Ollama model installed (e.g., llama3, tinyllama)

## Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
# or download the files directly
```

### 2. Install Python Dependencies

```bash
pip install streamlit requests beautifulsoup4 lxml
```

### 3. Install and Configure Ollama

Follow the official Ollama installation guide for your platform:
https://github.com/jmorganca/ollama

### 4. Pull an Ollama Model

```bash
ollama pull llama3
```

You can also use other models like `tinyllama`, `mistral`, etc.

## Setup

### 1. Verify Ollama Installation

```bash
ollama --version
ollama list  # Should show the models you have pulled
```

### 2. Verify Python Dependencies

```bash
python -c "import streamlit, requests, bs4"
```

## Usage

### 1. Run the Application

```bash
cd LLM-Global-News-Topic-Tracker
streamlit run news_tracker.py
```

### 2. Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

### 3. Using the Application

1. The main interface will load with a title "🌍 Global News Topic Tracker (Ollama LLM)"
2. Click the "Fetch Latest News" button
3. The app will fetch the top 10 news items from Google News
4. For each news item, the application will display:
   - The headline
   - A link to read more
   - An AI-generated summary using your local Ollama model
5. Wait for all summaries to generate (there's a spinner for each item)

## Architecture

The application follows a simple client-server architecture:

```
[User Browser] 
     ↓
[Streamlit Server]
     ↓
[news_tracker.py]
     ↓
[Google News RSS Feed] → [BeautifulSoup Parser]
     ↓
[Ollama Local API] ← → [AI Model (e.g. llama3)]
     ↓
[Results Displayed]
```

### Components:

- **Streamlit UI**: Provides the web interface
- **RSS Fetcher**: Retrieves Google News RSS feed
- **HTML Parser**: Uses BeautifulSoup to extract news items
- **Ollama Interface**: Communicates with local Ollama instance
- **Model Integration**: Uses subprocess to interact with Ollama

## Code Structure

### Functions in news_tracker.py:

- `summarize_with_ollama(text, model="llama3")`
  - Takes a text string and model name
  - Uses subprocess to run Ollama with the provided text
  - Returns the model's response or an error message

- `fetch_google_news()`
  - Fetches Google News RSS feed
  - Parses the XML response with BeautifulSoup
  - Returns the top 10 news items with titles and links

- Streamlit UI components
  - Sets up the web interface
  - Handles the "Fetch Latest News" button click
  - Displays results with headlines, links, and AI summaries

## Dependencies

### Python Dependencies:
- `streamlit`: Web framework for the UI
- `requests`: HTTP requests to fetch RSS feed
- `beautifulsoup4`: HTML/XML parsing
- `lxml`: XML parser for BeautifulSoup

### External Dependencies:
- `ollama`: Local AI model runner

### System Dependencies:
- Python 3.7+
- Internet connection to fetch RSS feed

## Troubleshooting

### Common Issues:

**1. Error calling Ollama or subprocess fails**
- Verify Ollama is installed and running: `ollama --version`
- Check if the model is pulled: `ollama list`
- Verify the model name in the code matches one in your list

**2. Streamlit won't start**
- Make sure all Python dependencies are installed
- Run: `pip install streamlit requests beautifulsoup4 lxml`

**3. RSS feed fetch fails**
- Check your internet connection
- Verify the RSS URL is still valid
- Google News RSS might have regional restrictions (the current URL targets India)

**4. Import errors**
- Make sure you're using Python 3.7+
- Reinstall dependencies: `pip install --upgrade streamlit requests beautifulsoup4 lxml`

### Running with Different Models

To use a different model, change the default parameter in the `summarize_with_ollama` function:
```python
def summarize_with_ollama(text, model="tinyllama"):  # or "mistral", etc.
```

## License

This project is open source and available under the MIT License.