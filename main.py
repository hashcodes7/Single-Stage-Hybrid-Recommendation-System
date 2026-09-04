from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from api.routes import router as api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    yield
    logger.info("Shutting down application...")
    
app = FastAPI(
    title="Single Stage Hybrid Recommendation System API",
    description="Unified API for fetching recommendations and training models",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}
