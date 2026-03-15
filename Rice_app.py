import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import gdown
import os
import cv2
import matplotlib.cm as cm

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="AI Rice Leaf Disease Detection Using Transfer Learning",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------
# HEADER
# ---------------------------

st.title("🌾 AI-Powered Rice Leaf Disease Detection")

st.markdown(
"""
This intelligent system uses **Deep Learning and Computer Vision**
to detect rice plant diseases from leaf images and recommend treatment.

AI is transforming agriculture by enabling **early disease detection and crop protection**.
"""
)

st.divider()

# ---------------------------
# MODEL CONFIG
# ---------------------------

MODEL_PATH = "rice_disease_model.h5"
FILE_ID = "1vnAI0WWQ2t7bBLfre37XFt_N4AQP5wV9"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
IMG_SIZE = 224


# ---------------------------
# DOWNLOAD MODEL
# ---------------------------

def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading trained AI model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return MODEL_PATH


# ---------------------------
# LOAD MODEL
# ---------------------------

@st.cache_resource
def load_model():

    path = download_model()
    model = tf.keras.models.load_model(path)

    return model


model = load_model()


# ---------------------------
# CLASSES
# ---------------------------

classes = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut",
    "Non Rice Leaf"
]


# ---------------------------
# DISEASE INFORMATION
# ---------------------------

disease_info = {

"Bacterial Leaf Blight":{
"description":"A bacterial infection causing yellowing and drying of rice leaves.",
"treatment":[
"Apply Streptomycin spray",
"Use copper fungicide",
"Plant resistant rice varieties"
]},

"Brown Spot":{
"description":"A fungal disease causing brown lesions on rice leaves.",
"treatment":[
"Apply Mancozeb fungicide",
"Improve soil fertility",
"Use treated seeds"
]},

"Leaf Smut":{
"description":"A fungal disease producing black spots on rice leaves.",
"treatment":[
"Remove infected leaves",
"Apply copper fungicide",
"Maintain field hygiene"
]},

"Non Rice Leaf":{
"description":"The uploaded image does not appear to be a rice leaf.",
"treatment":[
"No treatment required"
]}
}


# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------

def preprocess(img):

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------

def predict(img):

    preds = model.predict(img)

    index = np.argmax(preds)

    label = classes[index]
    confidence = float(np.max(preds))

    return label, confidence, preds


# ---------------------------
# GRAD-CAM FUNCTION
# ---------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

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

    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# ---------------------------
# HEATMAP OVERLAY FUNCTION
# ---------------------------

def overlay_heatmap(heatmap, image):

    img = np.array(image.resize((224,224)))

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * 0.4 + img
    superimposed_img = np.uint8(superimposed_img)

    return heatmap_color, superimposed_img


# ---------------------------
# MAIN LAYOUT
# ---------------------------

col1, col2 = st.columns([2,1])

uploaded = col1.file_uploader(
"Upload Rice Leaf Image",
type=["jpg","png","jpeg"]
)

if uploaded:

    image = Image.open(uploaded)

    col1.image(image, caption="Uploaded Leaf Image")

    img = preprocess(image)

    label, confidence, preds = predict(img)

    col2.subheader("Prediction")

    col2.success(label)

    col2.metric("Confidence", f"{confidence*100:.2f}%")

    st.divider()

    # ---------------------------
    # GRAD-CAM VISUALIZATION
    # ---------------------------

    st.subheader("Explainable AI (Grad-CAM Visualization)")

    heatmap = make_gradcam_heatmap(
        img,
        model,
        last_conv_layer_name="block5_conv3"
    )

    heatmap_color, overlay = overlay_heatmap(heatmap, image)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(image, caption="Original Rice Leaf")

    with c2:
        st.image(heatmap_color, caption="Grad-CAM Heatmap")

    with c3:
        st.image(overlay, caption="Model Attention Region")

    st.divider()

    # ---------------------------
    # DISEASE DESCRIPTION
    # ---------------------------

    st.subheader("Disease Description")

    st.write(disease_info[label]["description"])

    st.subheader("Recommended Treatment")

    for t in disease_info[label]["treatment"]:
        st.write("✔", t)

    st.divider()

    # ---------------------------
    # PREDICTION PROBABILITY
    # ---------------------------

    st.subheader("Prediction Probability")

    df = pd.DataFrame({
        "Disease":classes,
        "Probability":preds[0]
    })

    st.bar_chart(df.set_index("Disease"))


# ---------------------------
# ABOUT PROJECT
# ---------------------------

st.divider()

st.header("About This AI System")

st.markdown("""
This project uses **Deep Learning models trained on rice leaf images**
to automatically detect plant diseases.

### Models Used

• VGG16 (Transfer Learning)  
• ResNet50  
• MobileNetV2  
• Custom Convolutional Neural Network  

The **VGG16 model achieved the best performance** and is used in this application.
""")


# ---------------------------
# AI IN AGRICULTURE
# ---------------------------

st.header("AI in Agriculture")

st.markdown("""
Artificial Intelligence is revolutionizing agriculture by enabling
early detection of crop diseases using computer vision.

Benefits include:

• Early disease detection  
• Reduced pesticide usage  
• Increased crop yield  
• Smart farming decision support
""")


# ---------------------------
# AUTHOR
# ---------------------------

st.divider()

st.markdown("""
### Developed By

**Yuvraj Sharma**

Co-Author: **Ranadip Manna**
""")
