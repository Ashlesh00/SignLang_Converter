import streamlit as st
import cv2
import numpy as np
import base64
import requests

BACKEND_URL = "http://127.0.0.1:5000/predict"  # replace after deployment

st.title("Sign Language Live Translator")

frames_buffer = []

st.write("Show your sign in the camera...")

frame = st.camera_input("Record a short video")

if frame is not None:
    file_bytes = np.frombuffer(frame.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    _, buffer = cv2.imencode(".jpg", img)
    base64_frame = base64.b64encode(buffer).decode('utf-8')

    frames_buffer.append(base64_frame)

    if len(frames_buffer) == 30:
        st.write("⏳ Sending to model...")

        payload = {"frames": frames_buffer}
        res = requests.post(BACKEND_URL, json=payload)

        if res.status_code == 200:
            output = res.json()
            st.success(f"**Prediction:** {output['english']} ({output['gloss']})")
        else:
            st.error("Error from backend")

        frames_buffer = []
