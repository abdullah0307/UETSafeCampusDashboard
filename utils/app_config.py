from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
_CONFIG_CACHE: dict | None = None
_CONFIG_MTIME_NS: int | None = None


def load_app_config() -> dict:
    global _CONFIG_CACHE, _CONFIG_MTIME_NS

    current_mtime_ns = CONFIG_PATH.stat().st_mtime_ns
    if _CONFIG_CACHE is not None and _CONFIG_MTIME_NS == current_mtime_ns:
        return _CONFIG_CACHE

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    _CONFIG_CACHE = config
    _CONFIG_MTIME_NS = current_mtime_ns
    return config


def get_application_config(app_name: str) -> dict:
    config = load_app_config()
    applications = config.get("applications", {})
    app_config = applications.get(app_name)
    if not isinstance(app_config, dict):
        raise KeyError(f"Missing config for application: {app_name}")
    return app_config
