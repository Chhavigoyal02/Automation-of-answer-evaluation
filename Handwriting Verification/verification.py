import cv2
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st

class HandwritingVerification:
    def __init__(self):
        self.label_encoder = joblib.load('label_encoder.pkl')

    def predict_writer(self, image_path):
        model = tf.keras.models.load_model('trained_model.h5')
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        prediction = model.predict(img)
        predicted_id = np.argmax(prediction)
        predicted_label = self.label_encoder.inverse_transform([predicted_id])
        return predicted_label[0]

    def main(self, image_path, reference_id):
        predicted_id = self.predict_writer(image_path)
        if predicted_id == reference_id:
            return "Handwriting matches the student ID"
        else:
            return "Handwriting does not match the student ID"

def verification_interface():
    st.header("Handwriting Verification")
    uploaded_file = st.file_uploader("Upload a handwriting sample", type=["jpg", "png"])
    student_id = st.text_input("Enter the student ID")
    if st.button("Verify Handwriting"):
        if uploaded_file is not None and student_id:
            verification = HandwritingVerification()
            image_path = uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            result = verification.main(image_path, student_id)
            st.write(result)
        else:
            st.warning("Please upload an image and enter the student ID.")
