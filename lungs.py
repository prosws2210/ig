import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st

model_path = r'Lungs_Model.h5'

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

classes = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score

def get_lung_prediction(image):
    model = load_model()
    img_array = preprocess_image(image)
    score = predict_image(model, img_array)
    predicted_class_index = np.argmax(score)
    if predicted_class_index < len(classes):
        return f"Prediction: {classes[predicted_class_index]}", f"Confidence: {np.max(score) * 100:.2f}%"
    else:
        return "Error: Predicted class index is out of range.", ""
