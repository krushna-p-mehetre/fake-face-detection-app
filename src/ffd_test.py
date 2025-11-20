import numpy as np
import cv2
from keras.models import load_model
import os

# Model path (your .keras model)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "resnet_model_converted.keras")

# Test image path 
TEST_IMG_PATH = "E:\Krushna\Enggineering\BE\Project work\project\test_data\real.jpg"

# Load model
classifier = load_model(MODEL_PATH)

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

img = preprocess(TEST_IMG_PATH)

pred = classifier.predict(img)[0][0]

print("\n==============================")
print("🔍 Model Prediction Output")
print("==============================")
print(f"Prediction Score = {pred:.4f}")

if pred > 0.5:
    print("❌ Fake Image Detected")
else:
    print("✅ Real Image Detected")
