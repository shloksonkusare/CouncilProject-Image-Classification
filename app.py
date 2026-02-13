import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime
import subprocess

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Civic Issue Classifier", layout="centered")

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6
MODEL_PATH = "model/cnn_model.h5"

# ---------------------------
# LOAD CLASS NAMES
# ---------------------------
with open("model/class_names.txt") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# ---------------------------
# SAVE FUNCTION
# ---------------------------
def save_image_to_class(image, class_name):
    save_dir = os.path.join("NewData", class_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    save_path = os.path.join(save_dir, filename)

    image.save(save_path)
    return save_path

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------------------
# UI HEADER
# ---------------------------
st.title("🖼️ Civic Issue Classification App")

# ---------------------------
# IMAGE INPUT
# ---------------------------
st.markdown("### 📷 Capture Image")
camera_image = st.camera_input("Take a photo")

st.markdown("---")
st.markdown("### 📂 Or Upload an Image")
uploaded_file = st.file_uploader(
    "Upload image from device",
    type=["jpg", "png", "jpeg"]
)

image_file = camera_image if camera_image is not None else uploaded_file

# ---------------------------
# PREDICTION LOGIC
# ---------------------------
if image_file is not None:

    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Selected Image", use_column_width=True)

    resized_img = img.resize(IMG_SIZE)
    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    # ---------------------------
    # CONFIDENCE CHECK
    # ---------------------------
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Low confidence prediction. Needs review.")

    st.success(f"✅ Predicted Class: **{predicted_class}**")
    st.write(f"🎯 Confidence: **{confidence:.2f}**")

    # ---------------------------
    # SHOW CLASS PROBABILITIES
    # ---------------------------
    st.subheader("Class Probabilities")
    prob_dict = {
        CLASS_NAMES[i]: float(predictions[0][i])
        for i in range(len(CLASS_NAMES))
    }
    st.bar_chart(prob_dict)

    # ---------------------------
    # USER CONFIRMATION
    # ---------------------------
    st.markdown("---")
    st.subheader("Confirm Prediction")

    user_feedback = st.radio(
        "Is this prediction correct?",
        ("Yes", "No"),
        horizontal=True,
        key="feedback_radio"
    )

    if user_feedback == "Yes":
        if st.button("Save to Dataset", key="save_correct"):
            path = save_image_to_class(img, predicted_class)
            st.success(f"Image saved to: {path}")

    elif user_feedback == "No":
        correct_class = st.selectbox(
            "Select correct class:",
            CLASS_NAMES,
            key="correct_class"
        )

        if st.button("Save to Correct Class", key="save_corrected"):
            path = save_image_to_class(img, correct_class)
            st.success(f"Image saved to: {path}")

# ---------------------------
# ADMIN SECTION (OPTIONAL)
# ---------------------------
st.markdown("---")
st.subheader("🔧 Admin Panel")

if st.button("Retrain Model"):
    with st.spinner("Retraining model..."):
        subprocess.run(["python", "retrain.py"])
    st.success("Model retrained successfully!")
    st.cache_resource.clear()
    st.rerun()
