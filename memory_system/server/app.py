"""Optional server mode — run MemorySystem as a standalone API."""

from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
except ImportError:
    raise ImportError(
        "Server mode requires FastAPI. Install with: pip install memory_system[server]"
    )

from memory_system._client import MemorySystem
from memory_system.core.models import ChatRequest, ChatResponse, BotConfig
from memory_system.config.loader import load_all_configs
from memory_system.config.registry import BotRegistry


_contexts: dict[str, MemorySystem] = {}
_registry = BotRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configs_dir = (
        Path(app.state.configs_dir)
        if hasattr(app.state, "configs_dir")
        else Path("configs")
    )
    _registry.load(configs_dir)

    for bot in _registry.list_bots():
        _contexts[bot.bot_id] = MemorySystem.from_config(bot)
        print(f"  Loaded bot: {bot.bot_id} ({len(bot.intents)} intents)")

    print(f"Server ready with {len(_contexts)} bot(s)")
    yield
    _contexts.clear()


def create_server(configs_dir: str = "configs") -> FastAPI:
    app = FastAPI(
        title="MemorySystem Server",
        description="Intent-aware context management API",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.configs_dir = configs_dir

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        ctx = _contexts.get(request.bot_id)
        if not ctx:
            raise HTTPException(
                status_code=404, detail=f"Bot '{request.bot_id}' not found"
            )
        return await ctx.chat(request.message, session_id=request.session_id)

    @app.get("/api/bots")
    async def list_bots():
        return [
            {
                "bot_id": bot.bot_id,
                "bot_name": bot.bot_name,
                "intents": [i.name for i in bot.intents],
            }
            for bot in _registry.list_bots()
        ]

    @app.get("/api/bots/{bot_id}")
    async def get_bot(bot_id: str):
        bot = _registry.get(bot_id)
        if not bot:
            raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
        return bot.model_dump()

    @app.get("/api/bots/{bot_id}/analytics")
    async def get_analytics(bot_id: str):
        ctx = _contexts.get(bot_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
        return ctx.export_analytics()

    @app.get("/health")
    async def health():
        return {"status": "ok", "bots_loaded": len(_contexts)}

    return app


# Entry point: python -m memory_system.server.app
if __name__ == "__main__":
    import uvicorn

    app = create_server()
    uvicorn.run(app, host="0.0.0.0", port=8000)
