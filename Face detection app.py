# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 22:23:57 2025

@author: THINKPAD
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.write("Upload an image, pick rectangle color, adjust parameters, and detect faces.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

color_hex = st.color_picker("Pick rectangle color", "#00FF00")
color_bgr = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

min_neighbors = st.slider("minNeighbors", 1, 10, 3)
scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.1, 0.1)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color_bgr, 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    st.image(img_pil, caption=f"Detected {len(faces)} face(s)")

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Image with Faces",
        data=byte_im,
        file_name="detected_faces.jpg",
        mime="image/jpeg"
    )
else:
    st.write("Please upload an image.")