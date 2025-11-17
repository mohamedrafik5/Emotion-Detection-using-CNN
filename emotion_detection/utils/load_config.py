import os
import yaml
import cv2
import numpy as np
from tensorflow.keras.models import load_model


class LoadConfig:
    def __init__(self):
        # ✅ Make this point to the project root, not "emotion_detection"
        self.root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )

        # Path to config.yaml
        self.config_path = os.path.join(self.root_dir, "emotion_detection", "config", "config.yaml")

        # Load YAML
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Load model and cascade
        self._load_config()
        self.model, self.emotion_labels = self._model_loader()
        self.face_classifier = self._face_detector_loader()

    def _load_config(self):
        # Model path
        model_path = self.config["model_path"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.root_dir, "emotion_detection", model_path)
        self.model_path = os.path.normpath(model_path)

        # Face cascade path
        face_cascade_path = self.config.get("face_cascade_path")
        if not os.path.isabs(face_cascade_path):
            face_cascade_path = os.path.join(self.root_dir, "emotion_detection", face_cascade_path)
        self.face_cascade_path = os.path.normpath(face_cascade_path)

    def _model_loader(self):
        """Load the model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        model = load_model(self.model_path)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        return model, emotion_labels

    def _face_detector_loader(self):
        """Load Haar Cascade."""
        if not os.path.exists(self.face_cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found at {self.face_cascade_path}")
        return cv2.CascadeClassifier(self.face_cascade_path)
