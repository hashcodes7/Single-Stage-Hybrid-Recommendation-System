from update_user_embeddings import update_embeddings

# Dummy/Default dimensionality for embeddings if not specified
EMBEDDING_DIM = 3


def run_pipeline():
    print("--- Starting Recommendation Training Pipeline ---")
    
    # Update Embeddings based on recent events and save to Postgres
    update_embeddings()

    print("--- Training Pipeline Complete ---")

if __name__ == '__main__':
    run_pipeline()
