import yaml
from pathlib import Path

from app.core.models import BotConfig


def load_bot_config(file_path: Path) -> BotConfig:
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BotConfig(**data)


def load_all_configs(configs_dir: Path) -> dict[str, BotConfig]:
    configs = {}
    if not configs_dir.exists():
        return configs

    for file_path in configs_dir.glob("*.yaml"):
        config = load_bot_config(file_path)
        configs[config.bot_id] = config

    for file_path in configs_dir.glob("*.yml"):
        config = load_bot_config(file_path)
        configs[config.bot_id] = config

    return configs
