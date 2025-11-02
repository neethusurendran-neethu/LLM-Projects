# RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents (PDF or TXT) and ask questions about their content. The application uses Ollama for language models and Chroma for vector storage.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)

## Features

- Upload PDF or TXT documents
- Ask questions about the document content
- Context-aware responses based on the document
- Chat interface with user and bot message differentiation
- Memory-efficient model usage for smaller systems

## Architecture

The application follows a standard RAG architecture:

1. **Document Loading**: Loads PDF or TXT documents using LangChain loaders
2. **Text Splitting**: Splits documents into manageable chunks
3. **Embedding**: Creates vector embeddings using HuggingFace models
4. **Vector Storage**: Stores embeddings in ChromaDB
5. **Retrieval**: Retrieves relevant document chunks based on queries
6. **Generation**: Generates responses using Ollama language models

## Prerequisites

- Python 3.8 or higher
- Ollama running locally
- `tinyllama` model installed in Ollama (or other compatible models)
- ffmpeg (for audio processing, if using related features)

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install streamlit langchain-community langchain-huggingface langchain-ollama chromadb sentence-transformers PyPDF2
```

### 2. Install and Run Ollama

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and start the Ollama service:
```bash
ollama serve
```

### 3. Download the Required Model

```bash
ollama pull tinyllama
```

> **Note**: The application is configured to use `tinyllama` by default which requires less memory (638MB) and works well on systems with limited RAM.

### 4. Install ffmpeg (if needed)

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# CentOS/RHEL/Fedora
sudo dnf install ffmpeg
```

### 5. Run the Application

```bash
cd RAG_application
streamlit run app.py
```

## Usage

1. **Upload a Document**: Click "Browse files" to upload a PDF or TXT document
2. **Wait for Processing**: The application will process the document and create vector embeddings
3. **Ask Questions**: Type your questions in the chat input at the bottom of the page
4. **View Responses**: The bot will respond based on information from the uploaded document

### Chat Interface

- **User messages** appear on the right with a dark blue background
- **Bot responses** appear on the left with a gray background
- Responses are generated based only on the content of the uploaded document
- If the document doesn't contain the answer, the bot will respond with "I don't know from the document"

## Configuration

### Default Model Configuration

The application currently uses `tinyllama` as the default model:

```python
model_name = "tinyllama"
```

You can change this to other Ollama models by modifying the `model_name` variable in `app.py`, but consider memory requirements:
- `tinyllama`: ~638MB (recommended for low-memory systems)
- `mistral`: ~4.4GB (not recommended for systems with < 6GB RAM)
- `llama2`: ~3.8GB
- `gemma:2b`: ~1.7GB

### Chunk Settings

Documents are split into chunks with:
- `chunk_size=1000`: Maximum number of characters per chunk
- `chunk_overlap=200`: Number of overlapping characters between chunks

These settings balance context retention with processing efficiency.

## Troubleshooting

### Common Issues and Solutions

#### Memory Errors
- **Problem**: `model requires more system memory than is available`
- **Solution**: Use smaller models like `tinyllama` or `gemma:2b`

#### Ollama Not Found
- **Problem**: Connection errors to Ollama API
- **Solution**: Make sure Ollama is running with `ollama serve`

#### Document Upload Issues
- **Problem**: Issues with specific PDF formats
- **Solution**: Try converting to text or using different PDF software to re-save

#### Embedding Issues
- **Problem**: Slow processing of large documents
- **Solution**: The application creates a unique vector store for each session, which is automatically cleaned up

### Performance Tips

- For large documents, processing may take some time initially
- The vector database is stored temporarily in `.chroma/` directory for each session
- Uploaded files are temporarily stored in `Uploaded_Files/` directory

## Dependencies

### Python Libraries
- `streamlit`: Web interface
- `langchain-community`: Core RAG functionality
- `langchain-ollama`: Ollama integration
- `langchain-huggingface`: Embedding models
- `chromadb`: Vector storage
- `sentence-transformers`: Text embeddings
- `PyPDF2`: PDF processing

### External Dependencies
- `Ollama`: Local LLM service
- `ffmpeg`: Audio processing (optional)

## How It Works

The application implements a standard RAG pattern:

1. **Ingestion**: Documents are loaded, split into chunks, and converted to embeddings
2. **Storage**: Embeddings are stored in Chroma vector database
3. **Retrieval**: When a query is made, the system retrieves relevant chunks
4. **Generation**: The LLM generates an answer based on retrieved context

The prompt template ensures answers are based only on document content:

```
Answer the question using only the given context.
If the answer is not in the context, say "I don't know from the document."

Context:
{context}

Question: {question}
Answer:
```

## Limitations

- Responses are limited to information present in uploaded documents
- Large documents may take longer to process initially
- Vector stores are created per session and not persistent
- Requires local Ollama installation and running service

## License

This project is available as-is without any warranty. For personal and educational use.