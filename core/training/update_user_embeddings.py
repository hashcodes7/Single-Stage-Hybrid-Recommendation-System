import os
import numpy as np
import sqlite3
import constants
from core import database

def update_embeddings():
    print("Reading events from the database...")
    db_path = constants.EVENTS_DB_PATH if constants.EVENTS_DB_PATH else os.path.abspath(os.path.join(os.path.dirname(__file__), '../../events.db'))
    
    events = []
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            table_name = constants.EVENTS_TABLE_NAME
            c.execute(f"SELECT user_id, product_id FROM {table_name}")
            events = c.fetchall()
            conn.close()
        except Exception as e:
            print(f"Failed to read events: {e}")

    print(f"Found {len(events)} events.")
    
    # Group events by user_id
    user_event_history = {}
    for user_id, product_id in events:
        if user_id not in user_event_history:
            user_event_history[user_id] = []
        user_event_history[user_id].append(str(product_id))

    product_embeddings = database.get_product_embeddings()

    print("Precomputing user vectors based on events...")
    precomputed_user_vectors = {}
    
    for user_id, interacted_product_ids in user_event_history.items():
        # Fetch embeddings for all interacted products
        interacted_vectors = [
            product_embeddings[pid] 
            for pid in interacted_product_ids 
            if pid in product_embeddings
        ]
        
        if interacted_vectors:
            # User embedding is the mean of the interacted product embeddings
            user_vector = np.mean(interacted_vectors, axis=0)
            precomputed_user_vectors[user_id] = user_vector.reshape(1, -1)
        else:
            # If no valid products found, initialize randomly
            precomputed_user_vectors[user_id] = np.random.rand(1, constants.Embedding_dim)

    print("Saving updated User Vectors to Postgres database...")
    database.upsert_user_embeddings(precomputed_user_vectors)
