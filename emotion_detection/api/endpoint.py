from fastapi import FastAPI, UploadFile, File
from emotion_detection.core.model_invoking import EmotionRecognizer

app = FastAPI(title="Emotion Detection API")

recognizer = EmotionRecognizer()

@app.post("/predict_emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    boxes, labels = recognizer.predict_emotion(image_bytes)

    # ✅ Convert NumPy types to pure Python
    boxes = [tuple(map(int, box)) for box in boxes]  # ensures ints, not np.int32
    labels = [str(label) for label in labels]        # ensures strings

    return {"bounding_boxes": boxes, "labels": labels}


