"""Tests for JSON schema dataset generator."""

import json
import tempfile
from pathlib import Path

import jsonschema
import pytest

from rvu.data.json_schema_gen import generate_case, generate_dataset


class TestTierCoverage:
    """Each tier produces valid cases with correct structure."""

    @pytest.mark.parametrize("tier", [1, 2, 3, 4])
    def test_case_has_required_keys(self, tier):
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=tier, rng=rng)
        assert "case_id" in case
        assert "tier" in case
        assert "prompt" in case
        assert "schema" in case
        assert case["tier"] == tier
        assert case["case_id"] == 0

    @pytest.mark.parametrize("tier", [1, 2, 3, 4])
    def test_schema_is_valid_json_schema(self, tier):
        """Generated schemas themselves must be valid JSON Schemas."""
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=tier, rng=rng)
        schema = case["schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        # Validate that the schema is itself a valid JSON Schema
        jsonschema.Draft7Validator.check_schema(schema)

    @pytest.mark.parametrize("tier", [1, 2, 3, 4])
    def test_prompt_contains_schema(self, tier):
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=tier, rng=rng)
        schema_str = json.dumps(case["schema"], indent=2)
        assert schema_str in case["prompt"]
        assert "JSON:" in case["prompt"]

    def test_t1_flat_object(self):
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=1, rng=rng)
        schema = case["schema"]
        # All properties should be simple types
        for prop in schema["properties"].values():
            assert prop["type"] in ("string", "integer", "boolean")
        # 2-4 keys
        assert 2 <= len(schema["properties"]) <= 4

    def test_t2_has_nested_object(self):
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=2, rng=rng)
        schema = case["schema"]
        # At least one property should be type "object"
        has_nested = any(
            p.get("type") == "object" for p in schema["properties"].values()
        )
        assert has_nested

    def test_t3_has_array(self):
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=3, rng=rng)
        schema = case["schema"]
        has_array = any(
            p.get("type") == "array" for p in schema["properties"].values()
        )
        assert has_array

    def test_t4_has_enum_and_range(self):
        import random
        rng = random.Random(42)
        case = generate_case(case_id=0, tier=4, rng=rng)
        schema = case["schema"]
        has_enum = any(
            "enum" in p for p in schema["properties"].values()
        )
        has_range = any(
            "minimum" in p or "maximum" in p for p in schema["properties"].values()
        )
        assert has_enum
        assert has_range

    def test_t4_has_optional_fields(self):
        """T4 should have some fields not in required."""
        import random
        # Try multiple seeds to find one with optional fields
        found = False
        for seed in range(42, 100):
            rng = random.Random(seed)
            case = generate_case(case_id=0, tier=4, rng=rng)
            schema = case["schema"]
            all_keys = set(schema["properties"].keys())
            req_keys = set(schema["required"])
            if all_keys - req_keys:
                found = True
                break
        assert found, "T4 should sometimes produce optional fields"


class TestDeterminism:
    def test_same_seed_same_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "a.jsonl")
            path2 = str(Path(tmpdir) / "b.jsonl")

            generate_dataset(n=50, seed=42, output_path=path1)
            generate_dataset(n=50, seed=42, output_path=path2)

            with open(path1) as f1, open(path2) as f2:
                assert f1.read() == f2.read()

    def test_different_seed_different_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "a.jsonl")
            path2 = str(Path(tmpdir) / "b.jsonl")

            generate_dataset(n=50, seed=42, output_path=path1)
            generate_dataset(n=50, seed=99, output_path=path2)

            with open(path1) as f1, open(path2) as f2:
                assert f1.read() != f2.read()


class TestDatasetStructure:
    def test_200_cases_tier_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.jsonl")
            cases = generate_dataset(n=200, seed=42, output_path=path)

            assert len(cases) == 200

            tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for c in cases:
                tier_counts[c["tier"]] += 1
            assert tier_counts == {1: 50, 2: 50, 3: 50, 4: 50}

    def test_case_ids_unique(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.jsonl")
            cases = generate_dataset(n=200, seed=42, output_path=path)
            ids = [c["case_id"] for c in cases]
            assert len(ids) == len(set(ids))

    def test_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.jsonl")
            original = generate_dataset(n=20, seed=42, output_path=path)

            with open(path) as f:
                loaded = [json.loads(line) for line in f]

            assert len(loaded) == len(original)
            for orig, load in zip(original, loaded):
                assert orig == load

    def test_all_schemas_valid(self):
        """Every generated schema must itself be a valid JSON Schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.jsonl")
            cases = generate_dataset(n=200, seed=42, output_path=path)

            for case in cases:
                jsonschema.Draft7Validator.check_schema(case["schema"])
