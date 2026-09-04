from fastapi import APIRouter, BackgroundTasks
from core.recommendation.recommender import recommend
from core.training.trainer import run_pipeline

router = APIRouter()

@router.get("/recommendations/{user_id}", summary="Get Recommendations")
async def get_recommendations(user_id: int):
    """
    Get a list of recommendations for a given user.
    """
    try:
        recommendations = recommend(user_id=str(user_id), top_n=3)
        return {
            "user_id": user_id,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": str(e)}

@router.post("/train", summary="Trigger Training Pipeline")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers the training pipeline which updates embeddings and knn indexes.
    """
    background_tasks.add_task(run_pipeline)
    return {"status": "success", "message": "Training pipeline started in the background."}
