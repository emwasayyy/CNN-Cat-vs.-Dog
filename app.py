
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cat_vs_dog.h5')

# Define the image dimensions
img_height = 150
img_width = 150

st.title("Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = image.resize((img_height, img_width))
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)
    score = prediction[0]

    if score > 0.5:
        st.write(f"Prediction: Dog (Confidence: {score:.2f})")
    else:
        st.write(f"Prediction: Cat (Confidence: {1 - score:.2f})")
