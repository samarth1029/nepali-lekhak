import yaml
from pathlib import Path

class ConfigLoader:
    @staticmethod
    def load_config(path: str) -> dict:
        with open(Path(path), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)