from core.recommendation.get_user_vector import get_user_vector
from core.recommendation.get_rankings import get_rankings
from core import database

def recommend(user_id: str, top_n: int = 3):
    """
    Orchestrates the recommendation pipeline:
    1. Get User Representation
    2. Retrieve Candidates
    3. Create Rankings
    4. Return Top-N
    """
    # 1. Get User Representation
    user_vector = get_user_vector(user_id)
    
    # 2. Retrieve Candidates directly from pgvector
    candidates = database.retrieve_candidates_from_db(user_vector, n_candidates=top_n * 2)
    
    # 3. Create Rankings (Only Similarity)
    df_final = get_rankings(candidates)
    
    # 4. Return Top-N
    top_recommendations = []
    for _, row in df_final.head(top_n).iterrows():
        prod_id = row['product_id']
        top_recommendations.append({
            "item_id": prod_id,
            "similarity_score": round(row['score'], 5),
            "sim_rank": int(row['sim_rank'])
        })
        
    return top_recommendations
