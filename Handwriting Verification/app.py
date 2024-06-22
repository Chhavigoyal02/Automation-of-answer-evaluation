import streamlit as st
from auth import login, signup, check_user_role
from staff_module import staff_interface
from admin_module import admin_interface
from student_module import student_interface

def main():
    st.set_page_config(page_title="Handwriting Verification System", layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.session_state['role'] = ""

    if st.session_state['logged_in']:
        st.sidebar.subheader(f"Welcome, {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            logout()

        if st.session_state['role'] == 'admin':
            admin_interface()
        elif st.session_state['role'] == 'staff':
            staff_interface()
        elif st.session_state['role'] == 'student':
            student_interface()
    else:
        login_page()

def login_page():
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        username, role = login()
        if username:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.experimental_rerun()
    elif choice == "Sign Up":
        signup()

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.session_state['role'] = ""
    st.experimental_rerun()

if __name__ == "__main__":
    main()
