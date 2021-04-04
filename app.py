import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2
import os 
import keras
import tensorflow as tf
from bokeh.models.widgets import Div

basepath = os.path.dirname(__file__)
MODEL_PATH = 'model_final.hdf5'
file_path = os.path.join(basepath, MODEL_PATH)
print(file_path)


# Load your trained model
model = tf.keras.models.load_model(file_path)        # Necessary
nav = st.sidebar.radio(label = "Navigation Bar ",options = ["Cataract Prediction","Home","About","BMI Prediction","Disease Prediction","Heart Disease Prediction","Stroke Prediction"])
if nav == "Home":
    js = "window.open('https://mr-doctor-ml.herokuapp.com/')"  # New tab or window
    js = "window.location.href = 'https://mr-doctor-ml.herokuapp.com/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div) 
if nav == "About":
    js = "window.open('https://mr-doctor-ml.herokuapp.com/about')"  # New tab or window
    js = "window.location.href = 'https://mr-doctor-ml.herokuapp.com/about'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)     
if nav == "BMI Prediction":
    js = "window.open('https://mr-doctor-ml.herokuapp.com/BMI')"  # New tab or window
    js = "window.location.href = 'https://mr-doctor-ml.herokuapp.com/BMI'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div) 
if nav == "Disease Prediction":
    js = "window.open('https://mr-doctor-ml.herokuapp.com/symptoms')"  # New tab or window
    js = "window.location.href = 'https://mr-doctor-ml.herokuapp.com/symptoms'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div) 
if nav == "Heart Disease Prediction":
    js = "window.open('https://mr-doctor-ml.herokuapp.com/heart')"  # New tab or window
    js = "window.location.href = 'https://mr-doctor-ml.herokuapp.com/heart'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div) 
if nav == "Stroke Prediction":
    js = "window.open('https://mr-doctor-ml.herokuapp.com/stroke')"  # New tab or window
    js = "window.location.href = 'https://mr-doctor-ml.herokuapp.com/stroke'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div) 
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
col1, col2, col3 , col4, col5 = st.beta_columns(5)
with col3 :
    button = st.button('Predict',key = 1)
if button :
    if result[0][0] == 1 :
        st.error("You have chances of having a cataract! Take care of your eyes 😣")
    if result[0][0] == 0 :
        st.success("You are perfectly alright! Have a good day 😁")
