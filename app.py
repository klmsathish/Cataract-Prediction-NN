import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2
import os 
import keras

basepath = os.path.dirname(__file__)
MODEL_PATH = 'model_final.hdf5'
file_path = os.path.join(basepath, MODEL_PATH)
print(file_path)

# Load your trained model
model = keras.models.load_model(file_path)        # Necessary


print('Model loaded. Check http://127.0.0.1:5000/')
st.markdown("<h1 style='text-align: center;margin-top: -80px; color: blue;'>Automatic cataract detection</h1>", unsafe_allow_html=True)
st.write("This is a image classification web app to predict whether cataract from fundus images")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("You haven't uploaded an image file")
else:
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    image = Image.open(file)
    st.image(image, width = 250)
    
    image_size = 224
    try:
        image = cv2.cvtColor(np.array(image),cv2.IMREAD_COLOR)
        image = cv2.resize(image,(image_size,image_size))
    finally:
            print("done")
    val = np.array(image)
    vals = val.reshape(-1,image_size,image_size,3)    
    
    # Make prediction
    result = model.predict_classes(vals)

col1, col2, col3 = st.beta_columns(3)
col1, col2, col3 , col4, col5 = st.beta_columns(5)
with col3 :
    if st.button('Predict',key = 1):
        if result[0][0] == 1 :
            st.error("OMG! You have cataract")
        if result[0][0] == 0 :
            st.success("YAYYY !! You are very fine")
