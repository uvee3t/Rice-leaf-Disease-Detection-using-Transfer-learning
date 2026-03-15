import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Rice Leaf Disease Detection", layout="wide")

# -------------------------------
# Title
# -------------------------------

st.title("Rice Leaf Disease Detection using Deep Learning")
st.write("Upload a rice leaf image to detect disease and visualize infected region.")

# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():

    MODEL_PATH = "rice_model.h5"

    if not os.path.exists(MODEL_PATH):

        file_id = "YOUR_GOOGLE_DRIVE_MODEL_ID"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)

    return model


model = load_model()

# -------------------------------
# Class Names
# -------------------------------

class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut",
    "Healthy"
]

# -------------------------------
# Image Preprocessing
# -------------------------------

def preprocess_image(image):

    img = image.resize((224,224))
    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# -------------------------------
# GradCAM Function
# -------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3"):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# -------------------------------
# Disease Region Extraction
# -------------------------------

def extract_disease_region(img, heatmap):

    img = cv2.resize(img, (224,224))

    heatmap_resized = cv2.resize(heatmap, (224,224))

    heatmap_resized = np.uint8(255 * heatmap_resized)

    if len(heatmap_resized.shape) == 3:
        heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(heatmap_resized, 150, 255, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    disease_region = cv2.bitwise_and(img, img, mask=mask)

    return disease_region


# -------------------------------
# Upload Image
# -------------------------------

uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    img = np.array(image)

    st.image(image, caption="Uploaded Image", width=300)

    img_array = preprocess_image(image)

    # -------------------------------
    # Prediction
    # -------------------------------

    preds = model.predict(img_array)

    predicted_class = class_names[np.argmax(preds)]

    confidence = np.max(preds)

    st.subheader("Prediction")

    st.success(f"Disease: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

    # -------------------------------
    # GradCAM
    # -------------------------------

    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap_resized = cv2.resize(heatmap, (224,224))

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )

    superimposed_img = heatmap_colored * 0.4 + img

    # -------------------------------
    # Disease Region
    # -------------------------------

    disease_region = extract_disease_region(img, heatmap)

    # -------------------------------
    # Display Results
    # -------------------------------

    st.subheader("Model Visualization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img, caption="Original Leaf")

    with col2:
        st.image(superimposed_img.astype("uint8"), caption="GradCAM Heatmap")

    with col3:
        st.image(disease_region, caption="Detected Disease Region")
