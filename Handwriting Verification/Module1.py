import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Helper function to preprocess images
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

# Load and preprocess images
def load_images(folder_path):
    images = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = preprocess_image(img_path)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Define CNN model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Streamlit UI
st.title("Handwriting Verification System")

# Tab 1: New Data Upload
tab1, tab2, tab3 = st.tabs(["New Data Upload", "Handwriting Verification", "Supersub Identification"])

with tab1:
    st.header("Upload New Data and Train Model")
    data_folder = st.text_input("Enter the path to the dataset folder:")
    if st.button("Load and Train"):
        images, labels = load_images(data_folder)
        labels = pd.factorize(labels)[0]
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
        
        num_classes = len(np.unique(labels))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        
        model = create_model((128, 128, 1), num_classes)
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
        
        model.save("trained_model.h5")
        st.success("Model trained and saved successfully!")

# Tab 2: Handwriting Verification
with tab2:
    st.header("Verify Handwriting")
    model_path = st.text_input("Enter the path to the trained model:", "trained_model.h5")
    uploaded_image = st.file_uploader("Upload a handwritten image for verification")
    reference_id = st.text_input("Enter the reference Student ID:")
    
    if st.button("Verify"):
        if uploaded_image and reference_id:
            model = load_model(model_path)
            img = preprocess_image(uploaded_image)
            img = np.expand_dims(img, axis=0)
            
            pred = model.predict(img)
            predicted_id = np.argmax(pred)
            
            if predicted_id == int(reference_id):
                st.success("Handwriting matches the provided Student ID!")
            else:
                st.error("Handwriting does not match the provided Student ID.")

# Tab 3: Supersub Identification (Placeholder)
with tab3:
    st.header("Supersub Identification")
    st.write("This is a placeholder for future implementation.")
