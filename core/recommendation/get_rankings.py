import pandas as pd

def get_rankings(candidates):
    """
    Converts a list of candidate tuples (product_id, similarity) 
    into a ranked DataFrame.
    """
    similarity_data = []
    for product_id, similarity in candidates:
        similarity_data.append({'product_id': product_id, 'score': similarity})

    if not similarity_data:
        return pd.DataFrame(columns=['product_id', 'score', 'sim_rank'])

    df_sim = pd.DataFrame(similarity_data).sort_values('score', ascending=False).reset_index(drop=True)
    df_sim['sim_rank'] = df_sim.index + 1
    
    return df_sim
