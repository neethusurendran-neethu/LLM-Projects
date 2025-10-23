# LLM Multi-Modal Assistant

A comprehensive multi-modal assistant built with Streamlit that integrates text, image, and audio processing capabilities using Ollama and OpenAI Whisper.

## 🚀 Features

- **Text Chat**: Interactive text-based conversations using Ollama's language models
- **Image Understanding**: Visual analysis and description of uploaded images using the LLaVA model
- **Audio Transcription**: Speech-to-text transcription of audio files using OpenAI Whisper

## 🛠️ Tech Stack

- **Frontend**: Streamlit (UI framework)
- **Backend**: Python
- **LLM Integration**: Ollama API
- **Models Used**:
  - Llama3 (for text processing)
  - LLaVA (for image understanding)
  - Whisper (for audio transcription)
- **Audio Processing**: OpenAI Whisper
- **API Communication**: Requests library

## 📋 Prerequisites

Before running this application, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** running locally (accessible at `http://localhost:11434`)
3. **Llama3 model** installed in Ollama (`ollama pull llama3`)
4. **LLaVA model** installed in Ollama (`ollama pull llava`)
5. **Required Python libraries** (streamlit, requests, openai-whisper)

## 📦 Installation

1. Clone or download this repository to your local machine
2. Install the required Python dependencies:

```bash
pip install streamlit requests openai-whisper
```

3. Install the required models in Ollama:

```bash
ollama pull llama3
ollama pull llava
```

## 🔧 Configuration

The application uses the following default configuration:

- Ollama API endpoint: `http://localhost:11434/api/generate`
- Default text model: `llama3`
- Default image model: `llava`
- Whisper model used: `base` (can be changed to `small` or `medium` for higher accuracy)

## ▶️ Running the Application

1. Ensure Ollama is running locally:
   ```bash
   ollama serve
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## 📖 Usage

### Text Chat
1. Type your message in the text input field
2. Click "Send Text" to get a response from the Llama3 model

### Image Understanding
1. Upload an image file (JPG, JPEG, PNG)
2. Click "Analyze Image" to get a detailed description using the LLaVA model

### Audio Transcription
1. Upload an audio file (MP3, WAV, M4A)
2. Click "Transcribe Audio" to convert speech to text using Whisper

## 🏗️ Architecture

The application is structured in three main sections:
- **Text Chat**: Direct interaction with Ollama's text models
- **Image Analysis**: Image processing using LLaVA through Ollama API
- **Audio Transcription**: Local audio processing with Whisper

All communication with Ollama is handled through the `/api/generate` endpoint with appropriate payload formatting for different modalities.

## 🤖 Model Information

- **Llama3**: Advanced language model for text generation and understanding
- **LLaVA**: Large Language and Vision Assistant for multimodal tasks
- **Whisper**: Robust speech recognition model for audio transcription

## ⚠️ Notes

- Audio transcription happens locally using Whisper, which requires significant computational resources
- Image analysis requires the LLaVA model to be pulled in Ollama prior to use
- The application assumes Ollama is running on the default port (11434)
- Large audio files may take longer to transcribe

## 🔒 Privacy & Security

- All text and image processing is handled through your local Ollama instance
- Audio transcription is processed locally with Whisper
- No data is sent to external services, ensuring privacy of your content

## 🐛 Troubleshooting

- If you encounter issues with Ollama API calls, verify that Ollama is running and the required models are installed
- For audio transcription issues, ensure that your system has enough memory and computational resources
- Make sure file sizes are within the supported limits (200MB for images)

## 📞 Support

For support, please open an issue in the repository or contact the project maintainers.