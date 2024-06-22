import streamlit as st
import hashlib
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "postgresql://postgres:chhavi0206@localhost/project"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)
    role = Column(String)

Base.metadata.create_all(engine)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = session.query(User).filter_by(username=username, password=hash_password(password)).first()
        if user:
            return user.username, user.role
        else:
            st.error("Invalid username or password")
    return None, None

def signup():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["student", "staff", "admin"])
    if st.button("Sign Up"):
        if session.query(User).filter_by(username=username).first():
            st.error("Username already exists")
        else:
            new_user = User(username=username, password=hash_password(password), role=role)
            session.add(new_user)
            session.commit()
            st.success("Account created successfully")

def check_user_role(username):
    user = session.query(User).filter_by(username=username).first()
    if user:
        return user.role
    return None