import streamlit as st

import streamlit as st
import tensorflow as tf
import keras.utils as image
import numpy as np
from PIL import Image, ImageOps  # Streamlit works with PIL library very easily for Images
import cv2
import os

model_path = '.\\pretrainedModel\\pretrained_custom_model.h5py'


def prediction(savedModel, inputImage):
    test_image = image.load_img(
        inputImage,
        target_size=(300, 300))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = savedModel.predict(test_image)
    print("Predicted result", result)
    output = {0: 'caries', 1: 'gingivitis', 2: 'tooth Discoloration', 3: 'ulcer'}
    print("Output labels = ", output)
    print("output[np.argmax(result)] = ", output[np.argmax(result)])
    return output[np.argmax(result)]

def save_uploadedfile(uploadedfile, path):
    with open(path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    print("Saved File:{} to upload".format(uploadedfile.name))


st.title("Oral Diseases Prediction")
upload = st.file_uploader('Upload a food image')

if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)

    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # Color from BGR to RGB
    print("type of opencv", type(opencv_image))
    img = Image.open(upload)
    st.image(img, caption='Uploaded Image', width=300)
    if st.button('Predict'):
        # Load pretrained Model
        model = tf.keras.models.load_model(model_path)

        # model = tf.keras.models.load_model("pretrainedModel/Pretrained_CNNFoodClassificationModel.h5")

        path_dir = os.path.join(os.getcwd(), 'upload')
        print("path_dir =", path_dir)
        upload_path = os.path.join(path_dir, upload.name)
        print("upload_path=", upload_path)

        # Save uploaded file
        save_uploadedfile(upload, upload_path)

        # Prediction on uploaded image
        result = prediction(model, upload_path)
        st.title(result)

