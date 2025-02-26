
import psycopg2

def store_vectors_in_db(db_connection,embeddings, documents, file_name):
    if db_connection:
        try:
            cursor = db_connection.cursor()
            texts = [doc.page_content for doc in documents]
            vectors = embeddings.embed_documents(texts)

            for text, vector in zip(texts, vectors):
                cursor.execute(
                    "INSERT INTO items (embedding, text,file_name) VALUES (%s, %s,%s)",
                    (vector, text,file_name)
                )
            db_connection.commit()
            print("Vectors stored successfully.")
        except psycopg2.Error as e:
            print(f"Failed to store vectors: {e}")
            db_connection.rollback()
        finally:
            cursor.close()
