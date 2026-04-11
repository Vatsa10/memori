from fastapi import APIRouter, HTTPException

from app.bot_config.registry import registry
from app.core.models import BotConfig

router = APIRouter(prefix="/api", tags=["admin"])


@router.get("/health")
async def health():
    return {"status": "ok", "bots_loaded": len(registry.list_bots())}


@router.get("/bots")
async def list_bots():
    bots = registry.list_bots()
    return [
        {
            "bot_id": bot.bot_id,
            "bot_name": bot.bot_name,
            "intents": [i.name for i in bot.intents],
            "intent_model": bot.intent_model,
            "generation_model": bot.generation_model,
        }
        for bot in bots
    ]


@router.get("/bots/{bot_id}")
async def get_bot(bot_id: str):
    bot = registry.get(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    return bot.model_dump()


@router.post("/bots")
async def register_bot(config: BotConfig):
    registry.register(config)
    return {"status": "registered", "bot_id": config.bot_id}


@router.delete("/bots/{bot_id}")
async def remove_bot(bot_id: str):
    bot = registry.get(bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    registry.remove(bot_id)
    return {"status": "removed", "bot_id": bot_id}
