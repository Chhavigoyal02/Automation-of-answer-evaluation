import streamlit as st
from data_upload import data_upload_interface
from verification import verification_interface

def staff_interface():
    st.title("Staff Module")

    menu = ["Upload New Data and Train Model", "Handwriting Verification", "Supersub Identification"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload New Data and Train Model":
        data_upload_interface()
    elif choice == "Handwriting Verification":
        verification_interface()
    elif choice == "Supersub Identification":
        st.write("Placeholder for future functionality")
