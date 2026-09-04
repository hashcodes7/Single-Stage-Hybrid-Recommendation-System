import numpy as np
import constants
from core import database

def get_user_vector(user_id: str):
    """
    Fetches user vector from Postgres.
    Falls back to a random vector for unknown users (cold start).
    """
    user_vector = database.get_user_vector(user_id)
    if user_vector is not None:
        return user_vector
    
    # Fallback for unknown users (cold start): initialize randomly
    new_vector = np.random.rand(1, constants.Embedding_dim)
    # Store the newly initialized vector in Postgres
    database.upsert_user_embeddings({user_id: new_vector})
    return new_vector
