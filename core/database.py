import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
import constants

def get_pg_connection():
    """
    Connects to the PostgreSQL database using constants.
    """
    conn = psycopg2.connect(
        dsn=constants.PGVECTOR_DB_PATH,
        user=constants.PGVECTOR_DB_USERNAME,
        password=constants.PGVECTOR_DB_PASSWORD
    )
    register_vector(conn)
    return conn

def get_product_embeddings():
    """
    Fetches all product embeddings from the pgvector database.
    Returns a dictionary mapping product_id to its numpy array embedding.
    """
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        
        table = constants.PGVECTOR_PRODUCT_EMBEDDINGS_TABLE
        cur.execute(f"SELECT product_id, embedding FROM {table}")
        rows = cur.fetchall()
        
        product_embeddings = {}
        for row in rows:
            product_id = str(row[0])
            embedding = np.array(row[1])
            product_embeddings[product_id] = embedding
            
        cur.close()
        conn.close()
        return product_embeddings
    except Exception as e:
        print(f"Error fetching product embeddings: {e}")
        return {}

def upsert_user_embeddings(user_vectors):
    """
    Upserts user embeddings into the pgvector database.
    user_vectors: dict of {user_id: numpy_array}
    """
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        table = constants.PGVECTOR_USER_EMBEDDINGS_TABLE
        
        for user_id, vector in user_vectors.items():
            # Postgres upsert logic
            # Assuming table schema: (user_id INTEGER/TEXT PRIMARY KEY, embedding vector)
            # vector is converted to list for psycopg2/pgvector
            vec_list = vector.flatten().tolist()
            
            cur.execute(f"""
                INSERT INTO {table} (user_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET embedding = EXCLUDED.embedding;
            """, (user_id, vec_list))
            
        conn.commit()
        cur.close()
        conn.close()
        print("Successfully upserted user embeddings to Postgres.")
    except Exception as e:
        print(f"Error upserting user embeddings: {e}")

def get_user_vector(user_id):
    """
    Fetches a single user's vector from Postgres.
    Returns a numpy array or None if not found.
    """
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        table = constants.PGVECTOR_USER_EMBEDDINGS_TABLE
        
        cur.execute(f"SELECT embedding FROM {table} WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if row:
            return np.array(row[0]).reshape(1, -1)
        return None
    except Exception as e:
        print(f"Error fetching user vector: {e}")
        return None

def retrieve_candidates_from_db(user_vector, n_candidates=constants.recommended_candidates):
    """
    Performs KNN search directly in Postgres using pgvector.
    user_vector: numpy array of shape (1, dim)
    """
    try:
        conn = get_pg_connection()
        cur = conn.cursor()
        table = constants.PGVECTOR_PRODUCT_EMBEDDINGS_TABLE
        
        vec_list = user_vector.flatten().tolist()
        
        # Uses cosine distance `<=>` operator from pgvector
        cur.execute(f"""
            SELECT product_id, 1 - (embedding <=> %s::vector) AS similarity
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vec_list, vec_list, n_candidates))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        candidates = []
        for row in rows:
            candidates.append((str(row[0]), row[1]))
            
        return candidates
    except Exception as e:
        print(f"Error retrieving candidates from DB: {e}")
        return []
