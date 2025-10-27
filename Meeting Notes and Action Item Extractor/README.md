# Meeting Notes & Action Item Extractor

A Streamlit application that extracts summaries and action items from meeting transcripts using offline AI models.

## 🚀 Features

- **Offline Processing**: Uses locally loaded AI models, no internet required after initial setup
- **Meeting Summarization**: Automatically generates concise summaries of meeting content
- **Action Item Extraction**: Identifies and extracts action items from meeting notes using rule-based detection
- **Simple UI**: User-friendly interface for easy interaction

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **AI Model**: Facebook's BART-large-cnn for text summarization
- **Backend**: Python with Transformers library
- **Action Item Detection**: Rule-based approach using keyword matching

## 📋 Prerequisites

Before running this application, ensure you have:

1. **Python 3.7+** installed
2. **Required Python libraries** installed:
   - streamlit
   - transformers
   - torch (or tensorflow)
   - tokenizers

## 📦 Installation

1. Clone or download this repository to your local machine
2. Install the required Python dependencies with compatible versions:
   ```bash
   pip install streamlit==1.39.0 transformers==4.41.2 torch==2.3.1+cpu
   ```

3. The application will automatically download the BART model on first run (approx. 1.6GB)

## ▶️ Running the Application

1. Navigate to the project directory:
   ```bash
   cd "Meeting Notes and Action Item Extractor"
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`

## 📖 Usage

1. Open the application in your web browser
2. Paste your meeting transcript into the text area
3. Click the "Extract" button
4. View the generated summary and action items

## 🔧 How It Works

### Summarization
- Uses the `facebook/bart-large-cnn` model from Hugging Face Transformers
- Generates a summary with a maximum length of 60 words and minimum length of 20 words
- Uses beam search (do_sample=False) for consistent results

### Action Item Extraction
- Uses a simple rule-based approach
- Identifies lines containing keywords like "will" and "by" as potential action items
- Extracts these lines and displays them in a formatted list

## 🏗️ Code Structure

The application is contained in a single file (`app.py`) with the following components:

- **Model Loading**: BART summarization model is loaded at startup
- **UI Elements**: Streamlit components for input and output display
- **Processing Logic**: Summarization and action item extraction functions
- **Error Handling**: Validation for empty input

## ⚠️ Notes

- First run may take longer as it downloads the BART model (approx. 1.6GB)
- The action item detection is rule-based and may miss some items or include false positives
- The summarization quality depends on the BART model's ability to understand meeting context
- Application runs completely offline after the initial model download

## 🤖 Model Information

- **Model**: facebook/bart-large-cnn
- **Architecture**: BART (Bidirectional and Auto-Regressive Transformer)
- **Purpose**: Abstractive text summarization
- **Offline**: Works without internet after initial download

## 🔒 Privacy & Security

- All processing happens locally on your machine
- No data is sent to external servers
- Secure handling of your meeting content

## 🐛 Troubleshooting

**Issue**: Application takes a long time to start on first run
- **Solution**: This is normal - the BART model is ~1.6GB and needs to be downloaded

**Issue**: Out of memory error
- **Solution**: The BART model requires significant RAM; ensure your system has at least 8GB of RAM

**Issue**: "Tried to instantiate class '__path__._path', but it does not exist!" warning
- **Solution**: This is a known compatibility issue with certain PyTorch versions. The application will still work, but to resolve it, use the specific versions: `pip install streamlit==1.39.0 transformers==4.41.2 torch==2.3.1+cpu`

**Issue**: Poor summarization quality
- **Solution**: BART works best with well-structured text; consider cleaning up the meeting transcript before processing

## 📞 Support

For support, please open an issue in the repository or contact the project maintainers.

## 📄 License

This project is open source and available under the MIT License.