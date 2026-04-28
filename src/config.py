import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict):
    for key in ["model_dir", "figure_dir", "report_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
