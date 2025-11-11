import logging
import os
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


def _base_url() -> Optional[str]:
    url = os.environ.get("FIREBASE_DATABASE_URL")
    if not url:
        return None
    if not url.endswith("/"):
        url += "/"
    return url


def _auth_params() -> dict:
    token = os.environ.get("FIREBASE_DATABASE_AUTH")
    return {"auth": token} if token else {}


def _rules_url(base: str, key: str) -> str:
    return f"{base}prompt_rules/{key}.json"


def _normalize_rules(data) -> List[str]:
    rules: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                rule = item.strip()
                if rule:
                    rules.append(rule)
    elif isinstance(data, dict):
        for item in data.values():
            if isinstance(item, str):
                rule = item.strip()
                if rule:
                    rules.append(rule)
    return rules


def _fetch_rules_for_key(key: str) -> List[str]:
    base = _base_url()
    if not base:
        return []
    try:
        response = requests.get(_rules_url(base, key), params=_auth_params(), timeout=5)
        if response.status_code == 200:
            return _normalize_rules(response.json())
        if response.status_code not in (200, 404):
            logger.warning("Firebase rules fetch for %s failed: %s", key, response.text)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not fetch prompt rules for %s: %s", key, exc)
    return []


def get_prompt_rules(chat_id: Optional[str]) -> List[str]:
    """Return global + chat-specific prompt rules from Firebase (if configured)."""
    seen = set()
    ordered: List[str] = []
    for key in ("global", chat_id):
        if not key:
            continue
        for rule in _fetch_rules_for_key(key):
            if rule not in seen:
                seen.add(rule)
                ordered.append(rule)
    return ordered


def append_prompt_rule(target: str, rule: str) -> bool:
    """Append a prompt rule for the given target (chat id or 'global')."""
    base = _base_url()
    rule = (rule or "").strip()
    if not base or not rule:
        return False

    # Avoid duplicates by checking current rules first
    if rule in get_prompt_rules(target):
        return True

    try:
        response = requests.post(
            _rules_url(base, target),
            params=_auth_params(),
            json=rule,
            timeout=5,
        )
        if response.status_code == 200:
            return True
        logger.warning("Firebase rule append failed (%s): %s", response.status_code, response.text)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not append prompt rule: %s", exc)
    return False
