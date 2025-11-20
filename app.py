import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
tf.config.run_functions_eagerly(False)

# Load model only once 
@st.cache_resource
def load_ffd_model():
    return load_model("models/resnet_model_converted.keras")

model = load_ffd_model()

# Load Haarcascade face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Preprocessing Function
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit UI 
st.set_page_config(
    page_title="Fake vs Real Detector",
    page_icon="🕵️",
    layout="centered",
)

#  Custom CSS
st.markdown("""
<style>
    .title {
        font-size: 42px !important;
        font-weight: 800 !important;
        text-align: center;
        background: linear-gradient(90deg, #ff4b1f, #ff9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -20px;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #bdbdbd;
        margin-bottom: 30px;
    }
    .upload-label {
        font-size: 18px;
        font-weight: 500;
        padding-bottom: 8px;
    }
    .result-success {
        padding: 22px;
        border-radius: 12px;
        background-color: #e6ffe6;
        color: #0a8a0a;
        border: 2px solid #0a8a0a;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .result-error {
        padding: 22px;
        border-radius: 12px;
        background-color: #ffe6e6;
        color: #cc0000;
        border: 2px solid #cc0000;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

#  Header 
st.markdown("<div class='title'>🕵️ Fake vs Real Face Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload any face image and let the AI check if it's REAL or FAKE.</div>", unsafe_allow_html=True)

#  Upload Box 
uploaded_file = st.file_uploader(
    "📤 Drag & Drop an image or click to browse",
    type=["jpg", "jpeg", "png"]
)

# Prediction Logic
if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # FACE DETECTION
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("⚠️ No human face detected. Please upload a proper face image.")
    else:
        st.success(f"👤 Detected {len(faces)} face(s). Running deepfake check...")

        processed = preprocess_image(img)

        with st.spinner("🔍 AI analyzing image..."):
            pred = model.predict(processed)[0][0]

        st.subheader("📊 Prediction Score")
        st.write(f"**{pred:.4f}**")

        # RESULT DISPLAY 
        if pred > 0.5:
            st.markdown("<div class='result-error'>❌ Fake Image Detected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-success'>✅ Real Image Detected</div>", unsafe_allow_html=True)
