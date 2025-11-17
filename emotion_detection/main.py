import uvicorn
from emotion_detection.api.endpoint import app  # Import FastAPI app instance

if __name__ == "__main__":
    uvicorn.run("emotion_detection.api.endpoint:app", host="0.0.0.0", port=8000, reload=True)



# python -m emotion_detection.main