from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import os

app = Flask(__name__)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sign_model_final.h5")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "model", "labels.npy")

model = tf.keras.models.load_model(MODEL_PATH)
actions = np.load(LABELS_PATH)

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['video']
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    sequence = []
    cap = cv2.VideoCapture(file_path)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > 30:
                sequence = sequence[-30:]

    cap.release()
    os.remove(file_path)

    if len(sequence) < 30:
        return jsonify({'error': 'Video too short for prediction'})

    input_data = np.expand_dims(sequence[-30:], axis=0)
    res = model.predict(input_data)[0]
    predicted_word = actions[np.argmax(res)]

    return jsonify({'prediction': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
