
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="leaf_disease_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (example - replace with your actual class names)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

# App title
st.title("🌿 Leaf Disease Detection App")

# Image uploader
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf.', use_column_width=True)

    # Preprocess image
    img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    # Show prediction
    st.write(f"🩺 **Predicted Disease:** {class_names[prediction]}")
