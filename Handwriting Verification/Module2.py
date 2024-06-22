import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import psycopg2
import bcrypt

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        dbname="handwriting_verification",
        user="postgres",    # Replace with your actual username
        password="ayu1704@", # Replace with your actual password
        host="localhost",
        port="5432"
    )
    return conn

# Helper functions for database operations
def create_user(username, password, role):
    conn = get_db_connection()
    cur = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cur.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", 
                (username, hashed_password, role))
    conn.commit()
    cur.close()
    conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password, role FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[0].encode('utf-8')):
        return user[1]
    return None

def add_test_question(question, answer):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO test_questions (question, answer) VALUES (%s, %s)", 
                (question, answer))
    conn.commit()
    cur.close()
    conn.close()

def update_student_status(student_id, marks):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO student_status (student_id, marks) VALUES (%s, %s) ON CONFLICT (student_id) DO UPDATE SET marks = %s", 
                (student_id, marks, marks))
    conn.commit()
    cur.close()
    conn.close()

def get_student_status():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT student_id, marks FROM student_status")
    students = cur.fetchall()
    cur.close()
    conn.close()
    return students

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

if 'role' not in st.session_state:
    st.session_state['role'] = None

if st.session_state['role'] is None:
    auth_choice = st.radio("Select an option", ["Login", "Sign Up"])
    
    if auth_choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            role = authenticate_user(username, password)
            if role:
                st.session_state['role'] = role
                st.success(f"Logged in as {role}")
            else:
                st.error("Invalid credentials")
    
    if auth_choice == "Sign Up":
        st.subheader("Sign Up")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["admin", "staff", "student"])
        if st.button("Sign Up"):
            if new_username and new_password and new_role:
                create_user(new_username, new_password, new_role)
                st.success("User created successfully! Please log in.")

if st.session_state['role']:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Admin Module", "Staff Module", "Handwriting Verification", "Test QA Management", "Student Status", "Supersub Identification"])

    # Admin Module
    if st.session_state['role'] == 'admin':
        with tab1:
            st.header("Admin Module")
            admin_action = st.selectbox("Select Action", ["Add User", "Update User", "Delete User"])
            admin_username = st.text_input("Username")
            admin_password = st.text_input("Password", type="password")
            admin_role = st.selectbox("Role", ["admin", "staff", "student"])

            if admin_action == "Add User" and st.button("Submit"):
                create_user(admin_username, admin_password, admin_role)
                st.success("User added successfully")
            elif admin_action == "Update User" and st.button("Submit"):
                # Implement update user logic here
                st.info("Update user functionality not implemented")
            elif admin_action == "Delete User" and st.button("Submit"):
                # Implement delete user logic here
                st.info("Delete user functionality not implemented")

    # Staff Module
    if st.session_state['role'] in ['admin', 'staff']:
        with tab2:
            st.header("Staff Module")
            staff_action = st.selectbox("Select Action", ["Train Model", "Handwriting Verification", "Upload Test Questions", "Update User Role"])

            if staff_action == "Train Model":
                data_folder = st.text_input("Enter the path to the dataset folder:")
                if st.button("Load and Train"):
                    images, labels = load_images(data_folder)
                    labels = pd.factorize(labels)[0]
                    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
                    
                    num_classes = len(np.unique(labels))
                    y_train = to_categorical(y_train, num_classes)
                    y_test = to_categorical(y_test, num_classes)
                    
                    model = create_model((128, 128, 1), num_classes)
                    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
                    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
                    
                    model.save("trained_model.h5")
                    st.success("Model trained and saved successfully")

            if staff_action == "Handwriting Verification":
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

            if staff_action == "Upload Test Questions":
                uploaded_file = st.file_uploader("Upload CSV or Excel file with test questions and answers")
                if uploaded_file:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)

                    for index, row in df.iterrows():
                        add_test_question(row['question'], row['answer'])
                    st.success("Test questions uploaded successfully")

            if staff_action == "Update User Role":
                update_username = st.text_input("Enter the username to update role:")
                new_role = st.selectbox("New Role", ["admin", "staff", "student"])
                if st.button("Update Role"):
                    # Implement role update logic here
                    st.info("Update role functionality not implemented")

    # Handwriting Verification
    with tab3:
        st.header("Handwriting Verification")
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

    # Test QA Management
    if st.session_state['role'] in ['admin', 'staff']:
        with tab4:
            st.header("Test QA Management")
            uploaded_file = st.file_uploader("Upload CSV or Excel file with test questions and answers")
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)

                for index, row in df.iterrows():
                    add_test_question(row['question'], row['answer'])
                st.success("Test questions uploaded successfully")

    # Student Status
    if st.session_state['role'] in ['admin', 'staff', 'student']:
        with tab5:
            st.header("Student Status")
            students = get_student_status()
            st.write(pd.DataFrame(students, columns=["Student ID", "Marks"]))

    # Supersub Identification (Placeholder)
    with tab6:
        st.header("Supersub Identification")
        st.write("This is a placeholder for future implementation.")