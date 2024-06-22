import streamlit as st
from data_upload import data_upload_interface
from verification import verification_interface

def staff_interface():
    st.title("Staff Dashboard")
    tab1, tab2, tab3 = st.tabs(["New Data Upload", "Handwriting Verification", "Supersub Identification"])

    with tab1:
        data_upload_interface()

    with tab2:
        verification_interface()

    with tab3:
        st.header("Supersub Identification")
        st.write("Placeholder for future functionality")
