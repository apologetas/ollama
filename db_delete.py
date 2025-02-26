
import psycopg2

def delete_file(file_name, db_connection):
    if db_connection:
        try:
            cursor = db_connection.cursor()
            cursor.execute("DELETE FROM items WHERE file_name = %s",(file_name,))
            db_connection.commit()
            print(f"File '{file_name}' was deleted successfully")
        except psycopg2.Error as e:
            print(f"Failed to delete pdf file: {e}")
            db_connection.rollback()
        finally:
            cursor.close()
