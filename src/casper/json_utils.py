"""LLM-output parsing helpers.

Stage 1 generates JSON; Stage 4 generates `<answer>...</answer>` blocks.
Both stages share the loose JSON-from-text extractor.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

KEYWORD_KEYS: Tuple[str, ...] = (
    "core_concepts",
    "methodologies",
    "subjects_problems",
    "findings_impacts",
    "theoretical_framework",
    "quantitative_metrics",
    "contextual_background",
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PAREN_ID_RE = re.compile(r"[^(),]*?\((\d+)\)")
_ID_IN_ITEM_RE = re.compile(r"\((\d+)\)")


def extract_json_from_response(text: str) -> str:
    """Return the first parsable JSON object in ``text``, or ``text`` itself."""
    try:
        for match in _JSON_OBJECT_RE.findall(text):
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        return text
    except Exception:
        return text


def empty_keyword_json() -> str:
    return json.dumps({k: [] for k in KEYWORD_KEYS})


def validate_keyword_json(json_str: str, *, min_nonempty: int = 3) -> Tuple[bool, str]:
    """Check the 7-category keyword schema and require ``min_nonempty`` lists."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return False, f"JSON decode error: {exc}"

    for key in KEYWORD_KEYS:
        if key not in data:
            return False, f"Missing key: {key}"
        if not isinstance(data[key], list):
            return False, f"Key '{key}' is not a list"

    nonempty = sum(1 for k in KEYWORD_KEYS if data[k])
    if nonempty < min_nonempty:
        return False, f"Only {nonempty} non-empty categories (minimum {min_nonempty} required)"
    return True, "Valid JSON structure"


def keyword_json_score(json_str: str) -> int:
    """Number of non-empty keyword categories. 0 if unparseable."""
    try:
        data = json.loads(json_str)
    except Exception:
        return 0
    return sum(1 for k in KEYWORD_KEYS if isinstance(data.get(k), list) and data[k])


def count_keywords(json_str: str, *, default: int = 10) -> int:
    """Total keyword count across all categories. Falls back to ``default``."""
    try:
        data = json.loads(json_str)
    except Exception:
        return default
    total = 0
    for key in KEYWORD_KEYS:
        v = data.get(key)
        if isinstance(v, list):
            total += len(v)
    return total


def extract_summary_response(text: str) -> str:
    """Pull `{"response": "..."}` out of a Stage 1 summary completion."""
    json_str = extract_json_from_response(text)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        return json_str
    except Exception:
        return text


def parse_classification_answer(response: str) -> Optional[List[Dict[str, int]]]:
    """Parse `<answer>...</answer>` into ``[{"class_id": N}, ...]``.

    Returns ``[]`` for an explicit None answer, ``None`` if no answer tag exists
    or no class IDs could be extracted.
    """
    try:
        match = _ANSWER_RE.search(response)
        if not match:
            return None

        body = match.group(1).strip()
        if body.lower() == "none":
            return []

        body = body.strip("[]").strip()
        if not body:
            return []

        classes: List[Dict[str, int]] = []
        for m in _PAREN_ID_RE.finditer(body):
            classes.append({"class_id": int(m.group(1))})

        if classes:
            return classes

        # Fallback: comma-split with multiple format tolerance.
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            if item.isdigit():
                classes.append({"class_id": int(item)})
                continue
            stripped = item.strip("<>")
            inner = _ID_IN_ITEM_RE.search(stripped)
            if inner:
                classes.append({"class_id": int(inner.group(1))})
            else:
                print(f"Warning: Could not parse item: {item}")

        return classes if classes else None
    except Exception as exc:
        print(f"Error parsing answer: {exc}")
        return None
