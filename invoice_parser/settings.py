"""Settings changed in the app (the scanner folder), kept in data/settings.json."""
import json
import os

from . import config

SETTINGS_PATH = config.DATA_DIR / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(**values):
    data = load_settings()
    data.update(values)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, SETTINGS_PATH)
