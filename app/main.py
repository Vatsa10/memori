from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.bot_config.registry import registry
from app.core.intent_predictor import IntentPredictor
from app.core.pipeline import Pipeline
from app.providers.memory import InMemoryProvider
from app.api.routes_chat import router as chat_router, set_pipeline
from app.api.routes_admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load bot configs and initialize pipeline
    registry.load(settings.configs_path)
    print(f"Loaded {len(registry.list_bots())} bot config(s)")

    # Initialize intent predictor with embedding model
    predictor = IntentPredictor(embedding_model_name=settings.embedding_model)

    # Pre-compute intent embeddings for all bots
    for bot in registry.list_bots():
        predictor.precompute_intent_embeddings(bot)
        print(f"  - {bot.bot_id}: {len(bot.intents)} intents")

    # Create pipeline with InMemory provider (swap for Qdrant in production)
    memory = InMemoryProvider()
    pipeline = Pipeline(intent_predictor=predictor, memory_provider=memory)
    set_pipeline(pipeline)

    print("Pipeline ready")
    yield
    # Shutdown
    print("Shutting down")


app = FastAPI(
    title="Intent-Aware Context Management System",
    description="Smart prompt assembly — send what's needed, skip what's not.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(admin_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
