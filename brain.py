import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import streamlit as st

# Load the trained model
brain_model_path = r'best_brain_tumor_model.h5'
model = load_model(brain_model_path)

def preprocess_image(image):
    image = image.resize((150, 150))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0
    return image_array

def predict_tumor(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

def get_brain_prediction(image):
    result = predict_tumor(image)
    return "Tumor detected" if result > 0.5 else "No tumor detected"
