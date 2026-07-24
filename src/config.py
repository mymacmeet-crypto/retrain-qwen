from pathlib import Path
from functools import lru_cache
import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
