import streamlit as st
from auth import Session, User
import pandas as pd
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
        marking_scheme_interface()
        update_user_roles_interface()

def marking_scheme_interface():
    st.header("Upload Marking Scheme")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        if st.button("Save Marking Scheme"):
            df.to_csv("marking_scheme.csv", index=False)
            st.success("Marking scheme saved successfully")

def update_user_roles_interface():
    st.header("Update User Roles")

    session = Session()

    # Update User Role
    update_username = st.text_input("Username to update", key="staff_update_username")
    update_role = st.selectbox("New Role", ["student", "staff", "admin"], key="staff_update_role")
    if st.button("Update Role", key="staff_update_button"):
        user = session.query(User).filter_by(username=update_username).first()
        if user:
            user.role = update_role
            session.commit()
            st.success("User role updated successfully")
        else:
            st.error("User not found")

    session.close()
