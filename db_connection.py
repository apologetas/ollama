
import psycopg2
import os


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="ollama",
            user=os.getenv('POSTGRES_DATABASE_USERNAME'),
            password=os.getenv('POSTGRES_DATABASE_PASSWORD'),
            host="localhost",
            port="5555"
        )
        return conn
    except psycopg2.Error as e:
        print(e,"failed to connect to db")
        return None
