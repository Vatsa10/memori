from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM Providers
    groq_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Default Models
    intent_model: str = "groq/llama-3.1-8b-instant"
    generation_model: str = "groq/llama-3.3-70b-versatile"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "bot_knowledge"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Provider Preferences
    memory_store: str = "inmemory"  # "inmemory", "qdrant"
    graph_store: str = "inmemory"  # "inmemory", "neo4j"
    session_store: str = "inmemory"  # "inmemory", "redis"
    cache: str = "none"  # "none", "redis"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Bot Configs
    configs_dir: str = "configs"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def configs_path(self) -> Path:
        return Path(self.configs_dir)


settings = Settings()
