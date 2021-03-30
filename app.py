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

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')
st.write(
         # Image Classifier
         )
st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, width = 300)
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
    st.text(result[0])