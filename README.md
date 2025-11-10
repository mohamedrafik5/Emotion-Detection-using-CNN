🎭 Emotion Detection using CNN

A deep learning-based Emotion Detection System that identifies human emotions from facial expressions using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.
This project integrates a REST API for backend inference and a Streamlit interface for an interactive, real-time user experience.

📘 Overview

This project aims to detect emotions such as Happy, Sad, Angry, Disgust, Fear, Surprise, and Neutral from facial images.
Using OpenCV and a CNN model, it captures facial expressions and classifies them accurately, providing a lightweight and deployable emotion recognition system.

🚀 Features

🧠 Emotion detection using CNN trained on FER-2013 dataset

📷 Real-time emotion recognition using webcam feed

🌐 REST API endpoint for backend predictions

🖥️ Streamlit-based web interface for end-user interaction

⚙️ Configurable settings via YAML configuration file

🔧 Modular code structure for easy maintenance and scalability

🧩 Project Structure
Emotion Detection/
│
├── Api/
│   └── Endpoint.py             # REST API endpoint for emotion detection
│
├── config/
│   └── Config.yaml             # Configuration settings (paths, model, etc.)
│
├── core/
│   └── model_invoking.py       # Model loading and prediction logic
│
├── model/
│   ├── model.h5                # Trained CNN model
│   └── haar cascade.xml        # Haar Cascade for face detection
│
├── utils/
│   └── load_config.py          # Utility to load configuration
│
├── main.py                     # Entry point to launch the API server
├── streamlit.py                # Streamlit UI for emotion detection
└── README.md                   # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Emotion-Detection.git
cd Emotion-Detection

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the REST API
python main.py

5️⃣ Launch Streamlit Interface
streamlit run streamlit.py

🧠 Model Details

The CNN model is trained on the FER-2013 dataset, consisting of 48x48 grayscale images.

It classifies facial emotions into:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

The Haar Cascade classifier is used for face detection before prediction.

Model Highlights:

Multiple Conv2D + MaxPooling layers

Dropout for regularization

Dense layers with Softmax activation

Trained using Adam optimizer

🧩 Configurations

All configuration values (paths, parameters, etc.) are stored in config/Config.yaml.
You can modify this file to:

Change model path

Update detection parameters

Adjust API or Streamlit settings

🌐 API Usage

After running main.py, the REST API can be accessed locally:

Endpoint:

POST http://127.0.0.1:5000/predict


Sample JSON Request:

{
  "image": "base64_encoded_image_string"
}


Sample Response:

{
  "emotion": "Happy",
  "confidence": 0.97
}

🖥️ Streamlit App

You can interact with the model through an intuitive Streamlit UI:

streamlit run streamlit.py


Features:

Upload an image or use webcam

Detect emotions instantly

Display predicted label and confidence score

📊 Results
Metric	Value
Accuracy	~92%
Loss	<0.3
Dataset	FER-2013
Framework	TensorFlow / Keras
🔮 Future Enhancements

Implement Transfer Learning (VGGFace / ResNet50)

Add multi-face detection and emotion tracking

Deploy using Docker / Streamlit Cloud

Add voice-based emotion detection module

👨‍💻 Author

Mohamed Rafik A
📍 Chennai, Tamil Nadu
📧 mohameedrafik.a@gmail.com

🔗 LinkedIn

🙏 Acknowledgements

Dataset: FER-2013 on Kaggle

Frameworks: TensorFlow, Keras, OpenCV, Streamlit
