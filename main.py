import base64
import numpy as np

import streamlit as st

from PIL import Image
from keras.models import load_model


st.set_page_config(layout="wide", page_title="Leaf Disease")

st.write("## Leaf Disease Classifier")
st.write(
    ":dog: Try uploading an image."
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MODEL_PATH = "model"
CLASS_NAMES = ['bacterial_spot', 'early_blight', 'healthy_leaf', 'late_blight', 'leaf_mold', 'tomato_mosaic_virus']
model = load_model(MODEL_PATH)


def predict(img):
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)

    prediction = model.predict(
        np.array([
            img_array.astype("uint8"),
        ])
    )

    return prediction


def fix_image(upload):
    image = Image.open(upload)
    st.write("Original Image :camera:")
    st.image(image)

    prediction = predict(image)
    label = CLASS_NAMES[np.argmax(prediction)]
    st.write(f"Prediction: {label}")


my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
