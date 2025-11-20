🕵️‍♂️ Fake vs Real Face Detection (Deepfake Detector)

This project checks whether a face in an image is real or fake.
It uses a trained ResNet deep-learning model along with a clean and simple Streamlit web app.
The user can upload any photo, and the system will detect the face and classify it.

🚀 Features

Detects human faces in an image using OpenCV

Classifies each face as Real or Fake

Easy-to-use Streamlit web app

Clean and modern UI

Works on your local computer as well as online

Fast and optimized model

Supports common image formats (JPG, PNG, JPEG)

🧠 How It Works

The user uploads an image in the Streamlit app.

OpenCV finds the face in the image.

The face is resized and prepared for the model.

The trained ResNet model predicts if the face is Real or Fake.

The result is shown clearly on the screen.

🛠️ Tech Stack
Frontend / UI

Streamlit

Machine Learning

TensorFlow / Keras

ResNet50 (fine-tuned model)

GAN-based training (optional)

Computer Vision

OpenCV

Haarcascade face detection

Others

NumPy

Python 3.10
