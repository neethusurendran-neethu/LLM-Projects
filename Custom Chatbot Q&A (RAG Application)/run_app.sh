#!/bin/bash

cd "/home/seq_neethu/Desktop/LLM-Projects/Custom Chatbot Q&A (RAG Application)"

# Check if we need to set the OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "The application requires an OpenAI API key to run."
    echo "Please provide your OpenAI API key:"
    read -s -p "OpenAI API Key: " OPENAI_API_KEY
    echo ""
    export OPENAI_API_KEY
fi

echo "Starting the RAG Chatbot application..."
python3 -m streamlit run app.py