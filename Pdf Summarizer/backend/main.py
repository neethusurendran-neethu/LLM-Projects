# backend/main.py
from fastapi import FastAPI, UploadFile, File
import PyPDF2
from transformers import pipeline
import uvicorn

# Load summarization pipeline (uses Hugging Face model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

app = FastAPI()

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        return {"summary": "No text could be extracted from this PDF."}

    # Hugging Face summarization (can handle long text in chunks)
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summary_text = ""
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary_text += summary[0]['summary_text'] + " "

    return {"summary": summary_text.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
