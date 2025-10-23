from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow calls from frontend (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_msg = data["message"]
        model = data.get("model", "tinyllama:latest")  # Use tinyllama as default model

        logger.info(f"Received request for model: {model}, message: {user_msg}")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": user_msg, "stream": False}
        )
        
        logger.info(f"Ollama API response status: {response.status_code}")
        
        # Try to parse the JSON response regardless of status code
        # Ollama sometimes returns errors with status 500 but valid JSON
        try:
            response_data = response.json()
        except ValueError:
            # If response is not valid JSON
            logger.error(f"Ollama returned non-JSON response: {response.text}")
            return {"reply": f"❌ Response Error: {response.text}"}
        
        # Check if there's an error field in the response
        if "error" in response_data:
            logger.error(f"Ollama returned an error: {response_data['error']}")
            return {"reply": f"⚠️ Model Error: {response_data['error']}"}
        elif response.status_code == 200:
            # If status is 200 and no error field, return the response
            reply = response_data.get("response", "No response text received")
            logger.info(f"Returning reply: {reply}")
            return {"reply": reply}
        else:
            # If status is not 200 but no error field, return status info
            logger.error(f"Ollama API returned status code: {response.status_code}")
            return {"reply": f"⚠️ API Error: Status code {response.status_code}, Message: {response.text}"}
    except Exception as e:
        logger.error(f"Exception in chat endpoint: {str(e)}")
        return {"reply": f"❌ Connection Error: {str(e)}"}
