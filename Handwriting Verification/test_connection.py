from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:chhavi0206@localhost/project"

engine = create_engine(DATABASE_URL)
connection = engine.connect()
print("Connection successful!")
connection.close()
