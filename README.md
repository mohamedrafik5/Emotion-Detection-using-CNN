Emotion Detection Using CNN

This project aims to detect human emotions from facial expressions using a Convolutional Neural Network (CNN). The system is trained on the FER-2013 dataset (or a similar custom dataset) and can recognize multiple emotions such as Happy, Sad, Angry, Surprise, Neutral, Disgust, and Fear.

The project has two main components:

Model Training (Emotion_detection_model.ipynb) – building and training the CNN model

Real-Time Detection (detection.ipynb) – running real-time emotion detection using OpenCV and the trained model

🚀 Features

Detects emotions from live webcam or images

Uses deep learning (CNN) for high accuracy

Preprocessing includes image normalization and data augmentation

Real-time facial detection using OpenCV

Lightweight and easy to deploy

🧩 Project Structure
Emotion_Detection/
│
├── Emotion_detection_model.ipynb    # Model training and evaluation notebook
├── detection.ipynb                  # Real-time emotion detection notebook
├── model/                           # Saved CNN model (.h5)
├── haarcascade_frontalface_default.xml  # Face detection XML file
├── dataset/                         # Training and validation dataset (FER-2013 or custom)
└── README.md                        # Project documentation

⚙️ Tech Stack

Language: Python

Libraries: TensorFlow, Keras, NumPy, OpenCV

Model Architecture: Convolutional Neural Network (CNN)

Dataset: FER-2013 / Custom emotion dataset

📦 Installation

Clone the repository and install the dependencies:

git clone https://github.com/<your-username>/Emotion-Detection-CNN.git
cd Emotion-Detection-CNN
pip install -r requirements.txt


If you don’t have a requirements.txt, install manually:

pip install tensorflow keras opencv-python numpy

🧠 Model Training

Run the notebook Emotion_detection_model.ipynb to:

Load and preprocess the dataset

Define the CNN architecture

Train the model

Save the trained model (emotion_model.h5)

🎥 Real-Time Emotion Detection

Run detection.ipynb or a Python script to start detection:

python detection.py


It will:

Load the trained model

Capture webcam input

Detect faces using OpenCV Haar Cascade

Predict emotions in real time

😄 Detected Emotions

The model is capable of identifying:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

📊 Results

After training, the CNN model achieves high accuracy on both training and validation sets, demonstrating strong generalization for real-world faces.

🧪 Future Improvements

Use transfer learning (VGG16 / ResNet) for higher accuracy

Support for video emotion analysis

Web-based dashboard using Streamlit or Flask

👨‍💻 Author

Mohamed Rafik A
📍 Chennai, Tamil Nadu
📧 mohameedrafik.a@gmail.com

💼 LinkedIn : www.linkedin.com/in/mohamed-rafik-a-049436286
