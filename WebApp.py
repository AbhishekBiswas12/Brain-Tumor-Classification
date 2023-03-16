import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps


vgg16 = load_model('model (3).h5')

st.set_page_config(page_title='Brain Tumor Classification', layout='wide')
st.header("Brain Tumor Classification")

st.write("This is a Convolutional Neural Network Model Trained to recognise Brain Tumors from MRI scans")

file = st.file_uploader("Kindly upload an image here", type = ['jpg', 'png'])

def pred(img):
    size = (128, 128,)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    img = np.array(img)
    img = img/255.
    img_reshape = img[np.newaxis, ...]
    p = vgg16.predict(img_reshape)

    return p

if file is not None:
    img = Image.open(file)
    st.image(img, width = 300 )
    p = pred(img)
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    s = "This Image is Most Likely a : "+class_names[np.argmax(p)]
    st.success(s)