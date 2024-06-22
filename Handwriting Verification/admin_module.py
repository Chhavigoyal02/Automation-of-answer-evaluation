import streamlit as st
from auth import hash_password, Session, User

def admin_interface():
    st.title("Admin Dashboard")
    tab1, tab2 = st.tabs(["Manage Users", "Other Admin Tasks"])

    with tab1:
        manage_users_interface()

    with tab2:
        st.header("Other Admin Tasks")
        st.write("Placeholder for future admin functionalities")

def manage_users_interface():
    st.header("Manage Users")

    session = Session()

    # Add User
    st.subheader("Add New User")
    new_username = st.text_input("Username", key="add_username")
    new_password = st.text_input("Password", type="password", key="add_password")
    new_role = st.selectbox("Role", ["student", "staff", "admin"], key="add_role")
    if st.button("Add User"):
        if session.query(User).filter_by(username=new_username).first():
            st.error("Username already exists")
        else:
            new_user = User(username=new_username, password=hash_password(new_password), role=new_role)
            session.add(new_user)
            session.commit()
            st.success("User added successfully")

    # Update User Role
    st.subheader("Update User Role")
    update_username = st.text_input("Username to update", key="update_username")
    update_role = st.selectbox("New Role", ["student", "staff", "admin"], key="update_role")
    if st.button("Update Role"):
        user = session.query(User).filter_by(username=update_username).first()
        if user:
            user.role = update_role
            session.commit()
            st.success("User role updated successfully")
        else:
            st.error("User not found")

    # Delete User
    st.subheader("Delete User")
    delete_username = st.text_input("Username to delete", key="delete_username")
    if st.button("Delete User"):
        user = session.query(User).filter_by(username=delete_username).first()
        if user:
            session.delete(user)
            session.commit()
            st.success("User deleted successfully")
        else:
            st.error("User not found")

    # Display all users
    st.subheader("All Users")
    users = session.query(User).all()
    for user in users:
        st.write(f"Username: {user.username}, Role: {user.role}")

    session.close()
