import streamlit as st
from auth import login, signup, check_user_role
from data_upload import data_upload_interface
from verification import verification_interface
from staff_module import staff_interface
from admin_module import admin_interface

def main():
    st.title("Handwriting Verification System")
    
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Login":
        username, role = login()
        if username:
            st.sidebar.success(f"Logged in as {username} ({role})")
            if role == 'admin':
                admin_interface()
            elif role == 'staff':
                staff_interface()
            elif role == 'student':
                student_interface()
    elif choice == "Sign Up":
        signup()

if __name__ == "__main__":
    main()
