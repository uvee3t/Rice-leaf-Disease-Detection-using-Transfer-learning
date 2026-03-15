import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import gdown
import os

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

st.title("🌾 AI Based Rice Leaf Disease Detection System")

st.markdown("""
This application detects **rice leaf diseases using Deep Learning models**.

### Models Used
- AlexNet
- ResNet50
- MobileNetV2
- Ensemble CNN Model

The system also provides **Explainable AI using Grad-CAM visualization**.
""")

# -----------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------

MODEL_PATH = "rice_disease_model.h5"

FILE_ID = "1mgcTQlARjLqiJj8ZNnqHQGazIYUcgflh"

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

IMG_SIZE = 224

TRAINED_MODEL_ACCURACY = 0.96


# -----------------------------------------------------
# DOWNLOAD MODEL
# -----------------------------------------------------

def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading model from Google Drive..."):

            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return MODEL_PATH


# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------

@st.cache_resource
def load_model():

    path = download_model()

    model = tf.keras.models.load_model(path)

    return model


model = load_model()


# -----------------------------------------------------
# CLASS LABELS
# -----------------------------------------------------

classes = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut",
    "Healthy / Non Rice"
]


# -----------------------------------------------------
# DISEASE INFORMATION
# -----------------------------------------------------

disease_info = {

    "Bacterial Leaf Blight": {

        "description":
        "A bacterial disease caused by Xanthomonas oryzae causing yellowing of leaves.",

        "solution": [

            "Use resistant rice varieties",
            "Apply Streptomycin spray",
            "Avoid excessive nitrogen fertilizer"
        ]
    },

    "Brown Spot": {

        "description":
        "A fungal disease caused by Bipolaris oryzae producing brown lesions.",

        "solution": [

            "Apply Mancozeb fungicide",
            "Improve soil fertility",
            "Use treated seeds"
        ]
    },

    "Leaf Smut": {

        "description":
        "Fungal infection producing black spots on rice leaves.",

        "solution": [

            "Remove infected leaves",
            "Maintain field hygiene",
            "Apply copper fungicide"
        ]
    },

    "Healthy / Non Rice": {

        "description":
        "The image does not contain rice disease.",

        "solution": [
            "No treatment required"
        ]
    }

}


# -----------------------------------------------------
# IMAGE PREPROCESS
# -----------------------------------------------------

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image) / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------

def predict(img):

    preds = model.predict(img)

    index = np.argmax(preds)

    label = classes[index]

    confidence = float(np.max(preds))

    return label, confidence, preds


# -----------------------------------------------------
# AUTOMATIC LAST CONV LAYER DETECTION
# -----------------------------------------------------

def get_last_conv_layer(model):

    for layer in reversed(model.layers):

        if "conv" in layer.name.lower():

            return layer.name

    return None


# -----------------------------------------------------
# GRAD CAM
# -----------------------------------------------------

def make_gradcam_heatmap(img_array, model):

    last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()


# -----------------------------------------------------
# IMAGE UPLOAD
# -----------------------------------------------------

st.sidebar.header("Upload Image")

uploaded_file = st.sidebar.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg", "png", "jpeg"]
)


# -----------------------------------------------------
# MAIN APP
# -----------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    img = preprocess_image(image)

    label, confidence, preds = predict(img)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Uploaded Image")

        st.image(image, use_column_width=True)

    with col2:

        st.subheader("Prediction")

        st.success(label)

        st.metric("Confidence", f"{confidence*100:.2f}%")

        st.metric("Training Accuracy", f"{TRAINED_MODEL_ACCURACY*100:.2f}%")

    with col3:

        st.subheader("Disease Information")

        st.write(disease_info[label]["description"])

        st.write("### Solution")

        for s in disease_info[label]["solution"]:

            st.write("•", s)


    # -----------------------------------------------------
    # PREDICTION CHART
    # -----------------------------------------------------

    st.subheader("Prediction Probability")

    df = pd.DataFrame({

        "Disease": classes,

        "Probability": preds[0]

    })

    st.bar_chart(df.set_index("Disease"))


    # -----------------------------------------------------
    # GRADCAM VISUALIZATION
    # -----------------------------------------------------

    st.subheader("Explainable AI (Grad-CAM Visualization)")

    try:

        heatmap = make_gradcam_heatmap(img, model)

        heatmap = cv2.resize(heatmap, (224, 224))

        heatmap = np.uint8(255 * heatmap)

        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.cvtColor(
            np.array(image.resize((224, 224))),
            cv2.COLOR_RGB2BGR
        )

        overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Grad-CAM Heatmap")

            st.image(heatmap_color, channels="BGR")

        with col2:

            st.subheader("Grad-CAM Overlay")

            st.image(overlay, channels="BGR")

    except Exception as e:

        st.error("Grad-CAM visualization failed.")


# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------

st.markdown("---")

st.markdown("""
### Deep Learning Models Used

• AlexNet  
• ResNet50  
• MobileNetV2  
• Ensemble CNN Model  

These models were trained to improve rice disease detection accuracy.

Author: **Yuvraj Sharma**  
Co-Author: **Ranadip Manna**  
Guide: **Dr. Subhashis**
""")
