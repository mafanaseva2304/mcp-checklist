from functools import lru_cache
from pathlib import Path

import yaml

from settings.models import Settings

_DEFAULT_CONFIG = Path(__file__).parent.parent / "agent_config.yaml"


@lru_cache
def get_settings(config_path: str = str(_DEFAULT_CONFIG)) -> Settings:
    with open(config_path, "r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)
    return Settings(**yaml_config)


settings = get_settings()
