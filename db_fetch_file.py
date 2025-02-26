import psycopg2

def fetch_name(db_connection):
    if db_connection:
        try:
            cursor = db_connection.cursor()
            cursor.execute("SELECT DISTINCT file_name FROM items WHERE file_name IS NOT NULL")
            return cursor.fetchall()
            cursor.close()
            db_connection.close()
        except psycopg2.Error as e:
            print(f"Failed to fetch file name: {e}")
            db_connection.rollback()
        finally:
            cursor.close()