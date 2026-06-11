"""JSON schema adherence reward for the kill test.

Score regimes:
  0.0  — no parseable JSON found
  0.2 + 0.6 * (fraction of required keys with correct type)  — parseable but invalid
  1.0  — valid JSON that passes schema validation
"""

import json
import re
from typing import Optional

import jsonschema
from torch import Tensor

from .base import Reward
from .registry import register_reward


def extract_json_block(text: str) -> Optional[str]:
    """Extract the first balanced {...} block from text.

    Handles JSON embedded in prose, text before/after.
    Returns None if no balanced braces found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _count_correct_keys(obj: dict, schema: dict) -> tuple:
    """Count required keys present with correct type.

    Returns (correct_count, total_required).
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    if not required:
        return (0, 0)

    correct = 0
    for key in required:
        if key not in obj:
            continue
        if key not in properties:
            # Key is required but has no property spec — count as present
            correct += 1
            continue
        prop_schema = properties[key]
        expected_type = prop_schema.get("type")
        value = obj[key]

        if _type_matches(value, expected_type):
            correct += 1

    return (correct, len(required))


def _type_matches(value, expected_type: Optional[str]) -> bool:
    """Check if a Python value matches a JSON Schema type string."""
    if expected_type is None:
        return True
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True
    # In Python, bool is subclass of int — handle explicitly
    if expected_type == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


@register_reward("json_schema")
class JsonSchemaReward(Reward):
    """Reward for JSON schema adherence.

    Constructor takes the target schema dict.
    Counts its own invocations for accounting cross-checks.
    """

    def __init__(self, schema: dict) -> None:
        self.schema = schema
        self.call_count: int = 0

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        self.call_count += 1
        text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        return self.score_text(text)

    def score_text(self, text: str) -> float:
        """Score decoded text against the schema.

        Useful for direct text scoring without tokenizer round-trip.
        Does NOT increment call_count (use __call__ for that).
        """
        json_str = extract_json_block(text)
        if json_str is None:
            return 0.0

        try:
            obj = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return 0.0

        if not isinstance(obj, dict):
            return 0.0

        # Full validation
        try:
            jsonschema.validate(obj, self.schema)
            return 1.0
        except jsonschema.ValidationError:
            pass

        # Partial credit
        correct, total = _count_correct_keys(obj, self.schema)
        if total == 0:
            return 0.2
        fraction = correct / total
        return 0.2 + 0.6 * fraction
