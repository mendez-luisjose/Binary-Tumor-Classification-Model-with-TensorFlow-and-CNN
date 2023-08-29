import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.transform import resize
import tensorflow_hub as hub
from utils import set_background

set_background("./imgs/background.png")

#Put here the Model Path
MODEL_PATH = ''

width_shape = 256
height_shape = 256

header = st.container()
body = st.container()

def model_prediction(img, model):
    img_resize = resize(img, (width_shape, height_shape))
    X=tf.keras.applications.imagenet_utils.preprocess_input(img_resize*255)
    X = np.expand_dims(X,axis=0)

    X = np.vstack([X])
    
    preds = model.predict(X)
    percentage = preds * 100
    print(preds)

    if preds >= 0.75 :
        return "Malignant 🦠 | Percentage: " + "%.2f" % percentage + "%"
    else :
        return "Benign ✅"
    
with header :

    _, col1, _ = st.columns([0.15,1,0.1])
    col1.title("Tumor Classification 🏥")

    _, col2, _ = st.columns([0.2,1,0.2])
    col2.subheader("Binary Malignant - Benign Tumor Classification 🦠")

    _, col3, _ = st.columns([0.1,1,0.1])
    col3.image("./imgs/img.jpg", width=550)

    st.write("The Binary Model predicts if a given Tumor Image is either Malignant or Benign. The Model uses the CNN Architecture, built with TensorFlow.")

with body :
    _, col1, _ = st.columns([0.2,1,0.2])
    col1.subheader("Check It-out the Binary Malignant - Benign Tumor Classification Model 🔎!")

    model=''

    if model=='':
        model = tf.keras.models.load_model(
            (MODEL_PATH),
            custom_objects={'KerasLayer':hub.KerasLayer}, compile=False 
        )

    img = st.file_uploader("Upload a Tumor Image: ", type=["png", "jpg", "jpeg"])

    _, col2, _ = st.columns([0.2,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=500)
    

    if col5.button("Analyze Tumor"):
         prediction = model_prediction(image, model)
         _, col3, _ = st.columns([0.7,1,0.2])
         col3.header("Results ✅:")

         _, col4, _ = st.columns([0.1,1,0.1])
         col4.success("The Given Tumor is "  +  prediction)    




