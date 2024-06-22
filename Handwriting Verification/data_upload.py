import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

class NewDataUpload:
    def load_images(self, folder_path):
        images, labels = [], []
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))
                label = filename.split('_')[0]
                images.append(img)
                labels.append(label)
        images = np.array(images) / 255.0
        labels = np.array(labels)

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        joblib.dump(label_encoder, 'label_encoder.pkl')

        return images, labels, label_encoder

    def augment_data(self, images, labels):
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        class_counts = Counter(labels)
        for cls in class_counts:
            if class_counts[cls] < 2:
                class_images = images[labels == cls]
                num_to_generate = 2 - class_counts[cls]
                class_images_expanded = np.expand_dims(class_images, axis=-1)

                i = 0
                for _ in datagen.flow(class_images_expanded, batch_size=1, save_to_dir=None, save_prefix='aug', save_format='png'):
                    img = _.reshape(128, 128)
                    images = np.append(images, [img], axis=0)
                    labels = np.append(labels, [cls], axis=0)
                    i += 1
                    if i >= num_to_generate:
                        break
        return images, labels

    def main(self, folder_path):
        images, labels, label_encoder = self.load_images(folder_path)
        images, labels = self.augment_data(images, labels)
        class_counts = Counter(labels)
        print("Class distribution after augmentation:", class_counts)

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)
        return X_train, X_test, y_train, y_test, label_encoder

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val):
    input_shape = (128, 128, 1)
    num_classes = len(np.unique(y_train))
    model = create_model(input_shape, num_classes)
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save('trained_model.h5')

    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    
    return history

def data_upload_interface():
    st.header("Upload New Data and Train Model")
    folder_path = st.text_input("Enter the folder path for new data")
    if st.button("Load and Train"):
        data_upload = NewDataUpload()
        X_train, X_test, y_train, y_test, label_encoder = data_upload.main(folder_path)
        history = train_model(X_train, y_train, X_test, y_test)
        st.success("Model trained successfully")
