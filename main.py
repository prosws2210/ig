import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from brain import get_brain_prediction
from lungs import get_lung_prediction

# Paths to the model files
universal_model_path = r"universal.h5"


# Function to load a model with error handling
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None


# Load the universal model
universal_model = load_model(universal_model_path)


# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Streamlit app
st.title("Image Classifier App : Lungs or Brain")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img_array = preprocess_image(img)

    if universal_model is not None:
        # Make a prediction with the universal model
        prediction = universal_model.predict(img_array)

        # Determine if the image is of lungs or brain
        if prediction > 0.5:
            st.write("The image is classified as Lungs.")
            # Send the image to lungs.py
            lung_prediction, lung_confidence = get_lung_prediction(img)
            st.write(lung_prediction)
            st.write(lung_confidence)

        else:
            st.write("The image is classified as Brain.")
            # Send the image to brain.py
            brain_prediction = get_brain_prediction(img)
            st.write(brain_prediction)

    else:
        st.write("Universal model could not be loaded.")
