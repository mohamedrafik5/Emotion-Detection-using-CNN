import cv2
import numpy as np
from emotion_detection.utils.load_config import LoadConfig


class EmotionRecognizer:
    def __init__(self):
        config = LoadConfig()
        self.model = config.model
        self.emotion_labels = config.emotion_labels
        self.face_classifier = config.face_classifier

    def predict_emotion(self, image_bytes):
        """
        Takes image bytes as input and returns:
            - bounding boxes [(x, y, w, h), ...]
            - emotion labels [str, str, ...]
        """
        # Convert image bytes → NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Invalid image bytes or format not supported")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)

        boxes, labels = [], []

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = self.model.predict(roi, verbose=0)
            label = self.emotion_labels[np.argmax(prediction)]

            boxes.append((x, y, w, h))
            labels.append(label)

        return boxes, labels

# python -m emotion_detection.core.model_invoking

# if __name__ == "__main__":
#     recognizer = EmotionRecognizer()

#     # ✅ Path to your test image
#     image_path = r"C:\Users\ASUS\Downloads\WhatsApp Image 2025-09-22 at 13.20.51_8cb0311e.jpg"  # change this to your actual image path

#     # ✅ Read image as bytes (the function expects bytes)
#     with open(image_path, "rb") as f:
#         image_bytes = f.read()

#     # ✅ Run the emotion prediction
#     boxes, labels = recognizer.predict_emotion(image_bytes)
#     print(boxes,"------------",labels)

#     # ✅ Decode image for display
#     np_arr = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # ✅ Draw bounding boxes and emotion labels
#     for (x, y, w, h), label in zip(boxes, labels):
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#     # ✅ Display results — resized to fit screen
#     display_frame = frame.copy()
#     height, width = display_frame.shape[:2]
#     max_width = 1000  # adjust this if your screen is smaller/larger

#     if width > max_width:
#         scale = max_width / width
#         display_frame = cv2.resize(display_frame, (int(width * scale), int(height * scale)))

#     cv2.imshow("Emotion Recognition Result", display_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # ✅ Print results in console too
#     print("\n=== Emotion Recognition Results ===")
#     if boxes:
#         for (x, y, w, h), label in zip(boxes, labels):
#             print(f"Face at ({x}, {y}, {w}, {h}) → Emotion: {label}")
#     else:
#         print("No faces detected.")

