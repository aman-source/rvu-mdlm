"""JSON schema dataset generator for the kill test.

Four difficulty tiers:
  T1: flat object (2-4 keys, types string/int/bool)
  T2: nested object (one level of nesting)
  T3: arrays of objects
  T4: enums + integer ranges + required/optional mix

Each case = {case_id, tier, prompt, schema}.
Deterministic from seed.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Bounded noun/field pools for variety without explosion
NOUNS = [
    "person", "product", "vehicle", "recipe", "event",
    "company", "book", "city", "animal", "sensor",
    "task", "order", "message", "profile", "device",
]

STRING_FIELDS = [
    "name", "title", "description", "label", "category",
    "status", "color", "type", "code", "email",
]

INT_FIELDS = [
    "age", "count", "quantity", "price", "score",
    "width", "height", "weight", "duration", "level",
]

BOOL_FIELDS = [
    "active", "verified", "enabled", "public", "archived",
    "featured", "required", "premium", "visible", "completed",
]

ENUM_VALUES = {
    "status": ["active", "inactive", "pending", "archived"],
    "priority": ["low", "medium", "high", "critical"],
    "color": ["red", "green", "blue", "yellow", "black"],
    "size": ["small", "medium", "large", "extra-large"],
    "role": ["admin", "editor", "viewer", "guest"],
}


def _pick(rng: random.Random, pool: list, n: int) -> list:
    """Pick n unique items from pool."""
    return rng.sample(pool, min(n, len(pool)))


def _gen_t1(rng: random.Random) -> Tuple[Dict[str, Any], str]:
    """Flat object: 2-4 keys, types string/int/bool."""
    n_keys = rng.randint(2, 4)
    # Pick a mix of field types
    n_str = rng.randint(1, min(n_keys, 2))
    n_int = rng.randint(0, min(n_keys - n_str, 2))
    n_bool = n_keys - n_str - n_int

    fields: List[Tuple[str, str]] = []
    fields.extend((f, "string") for f in _pick(rng, STRING_FIELDS, n_str))
    fields.extend((f, "integer") for f in _pick(rng, INT_FIELDS, n_int))
    fields.extend((f, "boolean") for f in _pick(rng, BOOL_FIELDS, n_bool))
    rng.shuffle(fields)

    properties = {}
    for fname, ftype in fields:
        properties[fname] = {"type": ftype}

    required = [f for f, _ in fields]
    noun = rng.choice(NOUNS)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema, noun


def _gen_t2(rng: random.Random) -> Tuple[Dict[str, Any], str]:
    """Nested object: one level of nesting."""
    # Outer: 2-3 simple fields + 1 nested object
    n_outer = rng.randint(2, 3)
    outer_fields = _pick(rng, STRING_FIELDS, n_outer)
    outer_props = {f: {"type": "string"} for f in outer_fields}

    # Inner object: 2-3 fields
    n_inner = rng.randint(2, 3)
    inner_str = _pick(rng, [f for f in STRING_FIELDS if f not in outer_fields], min(1, n_inner))
    inner_int = _pick(rng, INT_FIELDS, n_inner - len(inner_str))
    inner_fields = [(f, "string") for f in inner_str] + [(f, "integer") for f in inner_int]
    inner_props = {f: {"type": t} for f, t in inner_fields}

    nested_name = rng.choice(["details", "metadata", "info", "config", "specs"])
    outer_props[nested_name] = {
        "type": "object",
        "properties": inner_props,
        "required": [f for f, _ in inner_fields],
        "additionalProperties": False,
    }

    required = list(outer_props.keys())
    noun = rng.choice(NOUNS)

    schema = {
        "type": "object",
        "properties": outer_props,
        "required": required,
        "additionalProperties": False,
    }
    return schema, noun


def _gen_t3(rng: random.Random) -> Tuple[Dict[str, Any], str]:
    """Arrays of objects."""
    # 1-2 simple top-level fields + 1 array of objects
    n_simple = rng.randint(1, 2)
    simple_fields = _pick(rng, STRING_FIELDS, n_simple)
    props = {f: {"type": "string"} for f in simple_fields}

    # Array item: 2-3 fields
    n_item = rng.randint(2, 3)
    item_str = _pick(rng, [f for f in STRING_FIELDS if f not in simple_fields], 1)
    item_int = _pick(rng, INT_FIELDS, n_item - len(item_str))
    item_fields = [(f, "string") for f in item_str] + [(f, "integer") for f in item_int]
    item_props = {f: {"type": t} for f, t in item_fields}

    array_name = rng.choice(["items", "entries", "records", "elements", "members"])
    props[array_name] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": item_props,
            "required": [f for f, _ in item_fields],
            "additionalProperties": False,
        },
        "minItems": 1,
    }

    required = list(props.keys())
    noun = rng.choice(NOUNS)

    schema = {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
    return schema, noun


def _gen_t4(rng: random.Random) -> Tuple[Dict[str, Any], str]:
    """Enums + integer ranges + required/optional mix."""
    # 3-5 fields total, mix of types
    n_fields = rng.randint(3, 5)
    props: Dict[str, Any] = {}
    all_fields: List[str] = []

    # At least one enum
    enum_key = rng.choice(list(ENUM_VALUES.keys()))
    props[enum_key] = {"type": "string", "enum": ENUM_VALUES[enum_key]}
    all_fields.append(enum_key)

    # At least one integer with range
    int_field = rng.choice(INT_FIELDS)
    min_val = rng.choice([0, 1, 10])
    max_val = rng.choice([100, 255, 1000])
    props[int_field] = {"type": "integer", "minimum": min_val, "maximum": max_val}
    all_fields.append(int_field)

    # Fill remaining with string/bool
    remaining = n_fields - 2
    extra_str = _pick(rng, [f for f in STRING_FIELDS if f not in all_fields], min(remaining, 2))
    for f in extra_str:
        props[f] = {"type": "string"}
        all_fields.append(f)
        remaining -= 1
    extra_bool = _pick(rng, BOOL_FIELDS, remaining)
    for f in extra_bool:
        props[f] = {"type": "boolean"}
        all_fields.append(f)

    # Required/optional mix: at least 2 required, rest optional
    n_required = rng.randint(2, max(2, len(all_fields) - 1))
    rng.shuffle(all_fields)
    required = all_fields[:n_required]
    noun = rng.choice(NOUNS)

    schema = {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
    return schema, noun


TIER_GENERATORS = {
    1: _gen_t1,
    2: _gen_t2,
    3: _gen_t3,
    4: _gen_t4,
}


def generate_case(
    case_id: int, tier: int, rng: random.Random
) -> Dict[str, Any]:
    """Generate a single (prompt, schema) case."""
    gen_fn = TIER_GENERATORS[tier]
    schema, noun = gen_fn(rng)

    schema_str = json.dumps(schema, indent=2)
    prompt = (
        f"Generate a JSON object describing a {noun}. "
        f"It must conform to this schema:\n{schema_str}\nJSON:"
    )

    return {
        "case_id": case_id,
        "tier": tier,
        "prompt": prompt,
        "schema": schema,
    }


def generate_dataset(
    n: int = 200,
    seed: int = 42,
    output_path: str = "data/kill_test_cases.jsonl",
) -> List[Dict[str, Any]]:
    """Generate n cases with equal tier distribution, write to JSONL.

    Args:
        n: Total number of cases.
        seed: Random seed for determinism.
        output_path: Path to write JSONL file.

    Returns:
        List of generated cases.
    """
    rng = random.Random(seed)
    n_tiers = 4
    per_tier = n // n_tiers
    remainder = n % n_tiers

    # Build tier list: equal distribution, extras go to later tiers
    tiers: List[int] = []
    for t in range(1, n_tiers + 1):
        count = per_tier + (1 if t <= remainder else 0)
        tiers.extend([t] * count)

    # Shuffle so tiers are interleaved
    rng.shuffle(tiers)

    cases = []
    for i, tier in enumerate(tiers):
        case = generate_case(case_id=i, tier=tier, rng=rng)
        cases.append(case)

    # Write JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    return cases
