import psycopg2

def retrieve_docs_from_db(db_connection, embeddings, query, l=4):
    if db_connection:
        try:
            cursor = db_connection.cursor()
            query_vector = embeddings.embed_documents([query])[0]
            cursor.execute(
                """
                SELECT text
                FROM items
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (query_vector, l)
            )

            results = cursor.fetchall()
            return [text for text, in results]

        except psycopg2.Error as e:
            print(f"Failed to retrieve similar documents: {e}")
        finally:
            cursor.close()
