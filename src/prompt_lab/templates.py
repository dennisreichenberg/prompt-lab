"""Named prompt template storage for prompt-lab."""

from __future__ import annotations

import json
from pathlib import Path

_STORE = Path.home() / ".prompt-lab" / "templates.json"


def _load() -> dict[str, str]:
    if not _STORE.exists():
        return {}
    try:
        return json.loads(_STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, str]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_template(name: str, prompt: str) -> None:
    data = _load()
    data[name] = prompt
    _save(data)


def get_template(name: str) -> str | None:
    return _load().get(name)


def all_templates() -> dict[str, str]:
    return _load()


def delete_template(name: str) -> bool:
    data = _load()
    if name not in data:
        return False
    del data[name]
    _save(data)
    return True
