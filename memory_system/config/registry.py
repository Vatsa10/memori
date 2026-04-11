from pathlib import Path

from memory_system.core.models import BotConfig
from memory_system.config.loader import load_all_configs


class BotRegistry:
    def __init__(self):
        self._bots: dict[str, BotConfig] = {}

    def load(self, configs_dir: Path):
        self._bots = load_all_configs(configs_dir)

    def get(self, bot_id: str) -> BotConfig | None:
        return self._bots.get(bot_id)

    def list_bots(self) -> list[BotConfig]:
        return list(self._bots.values())

    def register(self, config: BotConfig):
        self._bots[config.bot_id] = config

    def remove(self, bot_id: str):
        self._bots.pop(bot_id, None)


registry = BotRegistry()
