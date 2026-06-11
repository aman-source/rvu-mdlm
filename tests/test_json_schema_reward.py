"""Tests for JSON schema reward function."""

import json

import pytest
import torch
from unittest.mock import MagicMock

from rvu.rewards.json_schema import JsonSchemaReward, extract_json_block
from rvu.rewards.registry import get_reward


# ---------------------------------------------------------------------------
# extract_json_block
# ---------------------------------------------------------------------------

class TestExtractJsonBlock:
    def test_simple_object(self):
        assert extract_json_block('{"a": 1}') == '{"a": 1}'

    def test_embedded_in_prose(self):
        text = 'Here is the result: {"name": "Alice"} and some more text.'
        assert extract_json_block(text) == '{"name": "Alice"}'

    def test_nested_braces(self):
        text = '{"a": {"b": 1}}'
        assert extract_json_block(text) == '{"a": {"b": 1}}'

    def test_string_with_braces(self):
        text = '{"msg": "hello {world}"}'
        assert extract_json_block(text) == '{"msg": "hello {world}"}'

    def test_no_json(self):
        assert extract_json_block("no json here") is None

    def test_empty_string(self):
        assert extract_json_block("") is None

    def test_unbalanced_braces(self):
        assert extract_json_block('{"a": 1') is None

    def test_escaped_quotes(self):
        text = r'{"msg": "say \"hi\""}'
        result = extract_json_block(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["msg"] == 'say "hi"'

    def test_array_inside(self):
        text = '{"items": [1, 2, 3]}'
        assert extract_json_block(text) == '{"items": [1, 2, 3]}'


# ---------------------------------------------------------------------------
# Shared test schema
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Reward scoring
# ---------------------------------------------------------------------------

class TestRewardScoring:
    def setup_method(self):
        self.reward = JsonSchemaReward(schema=SIMPLE_SCHEMA)

    def test_perfect_json(self):
        text = '{"name": "Alice", "age": 30, "active": true}'
        assert self.reward.score_text(text) == 1.0

    def test_no_json_found(self):
        assert self.reward.score_text("no json here at all") == 0.0

    def test_truncated_json(self):
        assert self.reward.score_text('{"name": "Alice", "age":') == 0.0

    def test_valid_json_wrong_schema_no_keys(self):
        # Parseable but completely wrong keys
        text = '{"foo": "bar"}'
        score = self.reward.score_text(text)
        # 0 of 3 required keys correct: 0.2 + 0.6 * 0/3 = 0.2
        assert score == pytest.approx(0.2)

    def test_valid_json_partial_keys(self):
        # 1 of 3 required keys with correct type
        text = '{"name": "Alice", "foo": "bar"}'
        score = self.reward.score_text(text)
        # 1/3 correct: 0.2 + 0.6 * (1/3) = 0.4
        assert score == pytest.approx(0.2 + 0.6 * (1.0 / 3.0))

    def test_valid_json_wrong_type(self):
        # All keys present but age is string, not int
        text = '{"name": "Alice", "age": "thirty", "active": true}'
        score = self.reward.score_text(text)
        # 2/3 correct (name + active): 0.2 + 0.6 * 2/3 = 0.6
        assert score == pytest.approx(0.2 + 0.6 * (2.0 / 3.0))

    def test_all_keys_correct_type_but_extra_props(self):
        # Has all required keys with correct types but also extra properties
        text = '{"name": "Alice", "age": 30, "active": true, "extra": 1}'
        score = self.reward.score_text(text)
        # additionalProperties: false means schema validation fails
        # But all 3/3 keys correct: 0.2 + 0.6 * 1.0 = 0.8
        assert score == pytest.approx(0.8)

    def test_json_embedded_in_prose(self):
        text = 'The answer is: {"name": "Bob", "age": 25, "active": false} done.'
        assert self.reward.score_text(text) == 1.0

    def test_score_regimes_clean_separation(self):
        """0.0 for no JSON, [0.2, 0.8] for partial, 1.0 for valid."""
        # No JSON
        assert self.reward.score_text("hello") == 0.0

        # Partial (parseable but invalid)
        partial = self.reward.score_text('{"name": "A"}')
        assert 0.2 <= partial <= 0.8

        # Valid
        assert self.reward.score_text(
            '{"name": "A", "age": 1, "active": true}'
        ) == 1.0

    def test_boolean_not_counted_as_integer(self):
        """Python bool is subclass of int — must not count True as integer."""
        text = '{"name": "A", "age": true, "active": true}'
        score = self.reward.score_text(text)
        # age is bool not int: 2/3 correct
        assert score == pytest.approx(0.2 + 0.6 * (2.0 / 3.0))


# ---------------------------------------------------------------------------
# Call counting
# ---------------------------------------------------------------------------

class TestCallCounting:
    def test_call_count_increments(self):
        reward = JsonSchemaReward(schema=SIMPLE_SCHEMA)
        assert reward.call_count == 0

        tokenizer = MagicMock()
        tokenizer.decode.return_value = '{"name": "A", "age": 1, "active": true}'
        ids = torch.tensor([1, 2, 3])

        reward(ids, tokenizer)
        assert reward.call_count == 1

        reward(ids, tokenizer)
        reward(ids, tokenizer)
        assert reward.call_count == 3

    def test_call_uses_tokenizer(self):
        reward = JsonSchemaReward(schema=SIMPLE_SCHEMA)
        tokenizer = MagicMock()
        tokenizer.decode.return_value = '{"name": "A", "age": 1, "active": true}'
        ids = torch.tensor([10, 20, 30])

        score = reward(ids, tokenizer)
        tokenizer.decode.assert_called_once_with([10, 20, 30], skip_special_tokens=True)
        assert score == 1.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered(self):
        reward = get_reward("json_schema", schema=SIMPLE_SCHEMA)
        assert isinstance(reward, JsonSchemaReward)


# ---------------------------------------------------------------------------
# Edge cases with different schema types
# ---------------------------------------------------------------------------

class TestSchemaVariants:
    def test_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "info": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"],
                },
            },
            "required": ["name", "info"],
            "additionalProperties": False,
        }
        reward = JsonSchemaReward(schema=schema)
        assert reward.score_text('{"name": "A", "info": {"age": 5}}') == 1.0
        # Wrong nested type
        score = reward.score_text('{"name": "A", "info": "wrong"}')
        # info present but wrong type: 1/2 correct
        assert score == pytest.approx(0.2 + 0.6 * 0.5)

    def test_array_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
                    "minItems": 1,
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        reward = JsonSchemaReward(schema=schema)
        assert reward.score_text('{"items": [{"id": 1}]}') == 1.0
        # Empty array violates minItems
        score = reward.score_text('{"items": []}')
        # Key present with correct type (array): 1/1 = 0.8
        assert score == pytest.approx(0.8)

    def test_enum_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
            "required": ["status"],
            "additionalProperties": False,
        }
        reward = JsonSchemaReward(schema=schema)
        assert reward.score_text('{"status": "active"}') == 1.0
        # Wrong enum value
        score = reward.score_text('{"status": "unknown"}')
        # Key present, correct type (string): 1/1 = 0.8
        assert score == pytest.approx(0.8)
