# tests/test_extraction.py
import json
import pytest
from unittest.mock import MagicMock, patch
from backend.agent_logic import (
    create_expert_system_prompt,
    create_extraction_prompt,
    process_user_input_with_llm,
    PARAM_INFO,
)


class TestCreateExpertSystemPrompt:
    def test_returns_nonempty_string(self):
        prompt = create_expert_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 500

    def test_contains_is456_formula(self):
        prompt = create_expert_system_prompt()
        assert "5000" in prompt
        assert "fck" in prompt.lower() or "√fck" in prompt or "sqrt" in prompt.lower()

    def test_contains_contact_pressure_formula(self):
        prompt = create_expert_system_prompt()
        assert "q" in prompt
        assert "MPa" in prompt
        assert "Lx" in prompt or "contact" in prompt.lower()

    def test_contains_disambiguation_rules(self):
        prompt = create_expert_system_prompt()
        assert "ambiguous" in prompt.lower() or "disambiguation" in prompt.lower()
        assert "contact" in prompt.lower()

    def test_contains_node_array_rules(self):
        prompt = create_expert_system_prompt()
        assert "coarse" in prompt.lower()
        assert "medium" in prompt.lower()
        assert "fine" in prompt.lower()
        assert "linspace" in prompt.lower() or "evenly spaced" in prompt.lower() or "i×" in prompt or "i ×" in prompt

    def test_contains_load_patch_rules(self):
        prompt = create_expert_system_prompt()
        assert "corner" in prompt.lower()
        assert "interior" in prompt.lower()
        assert "x1" in prompt
        assert "y1" in prompt

    def test_contains_output_schema(self):
        prompt = create_expert_system_prompt()
        assert "needs_clarification" in prompt
        assert "understood" in prompt
        assert "computed" in prompt
        assert "load_cases" in prompt


class TestCreateExtractionPrompt:
    def test_includes_user_input(self):
        prompt = create_extraction_prompt(
            user_input="slab 5m x 3m",
            context="",
            current_asking=None,
            params={},
        )
        assert "slab 5m x 3m" in prompt

    def test_includes_current_asking_context(self):
        prompt = create_extraction_prompt(
            user_input="200 mm",
            context="",
            current_asking="t",
            params={},
        )
        assert "t" in prompt or "Slab Thickness" in prompt or "thickness" in prompt.lower()

    def test_includes_already_collected_params(self):
        prompt = create_extraction_prompt(
            user_input="M30",
            context="",
            current_asking="Emod",
            params={"a": 5000, "b": 3000},
        )
        assert "5000" in prompt
        assert "3000" in prompt


class TestProcessUserInputWithLLM:
    def _make_llm_mock(self, response_content: str):
        mock_response = MagicMock()
        mock_response.content = response_content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_valid_response_returns_computed_as_extracted_multiple(self):
        payload = {
            "needs_clarification": False,
            "clarification_question": None,
            "friendly_response": "Got it.",
            "use_default": False,
            "use_all_defaults": False,
            "conversation_only": False,
            "understood": {"slab_a_mm": 5000},
            "computed": {"a": 5000.0, "b": 3000.0},
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = process_user_input_with_llm("5m x 3m slab", mock_llm, "", {}, None)
        assert result["extracted_multiple"]["a"] == 5000.0
        assert result["extracted_multiple"]["b"] == 3000.0
        assert result["needs_clarification"] is False

    def test_needs_clarification_response(self):
        payload = {
            "needs_clarification": True,
            "clarification_question": "Is 5000x2000 the slab or contact area?",
            "friendly_response": "Is 5000x2000 mm the slab size or tyre contact?",
            "use_default": False,
            "use_all_defaults": False,
            "conversation_only": False,
            "understood": {},
            "computed": {},
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = process_user_input_with_llm("5000x2000", mock_llm, "", {}, None)
        assert result["needs_clarification"] is True
        assert result["extracted_multiple"] == {}

    def test_use_default_response(self):
        payload = {
            "needs_clarification": False,
            "clarification_question": None,
            "friendly_response": "Using default value.",
            "use_default": True,
            "use_all_defaults": False,
            "conversation_only": False,
            "understood": {},
            "computed": {},
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = process_user_input_with_llm("yes", mock_llm, "", {}, "a")
        assert result["use_default"] is True

    def test_malformed_json_returns_clarification(self):
        mock_llm = self._make_llm_mock("Sorry, I cannot help with that.")
        result = process_user_input_with_llm("test input", mock_llm, "", {}, None)
        assert result["needs_clarification"] is True
        assert mock_llm.invoke.call_count == 2  # initial attempt + one retry

    def test_markdown_fenced_json_is_parsed(self):
        payload = {
            "needs_clarification": False,
            "clarification_question": None,
            "friendly_response": "Captured.",
            "use_default": False,
            "use_all_defaults": False,
            "conversation_only": False,
            "understood": {},
            "computed": {"t": 250.0},
        }
        fenced = f"```json\n{json.dumps(payload)}\n```"
        mock_llm = self._make_llm_mock(fenced)
        result = process_user_input_with_llm("250mm thick", mock_llm, "", {}, "t")
        assert result["extracted_multiple"]["t"] == 250.0

    def test_timeout_returns_friendly_fallback(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("connection timeout")
        result = process_user_input_with_llm("test", mock_llm, "", {}, None)
        assert result["needs_clarification"] is True
        assert "friendly_response" in result

    def test_computed_values_include_node_arrays(self):
        payload = {
            "needs_clarification": False,
            "clarification_question": None,
            "friendly_response": "Got it.",
            "use_default": False,
            "use_all_defaults": False,
            "conversation_only": False,
            "understood": {"mesh_type": "coarse", "slab_a_mm": 3500},
            "computed": {
                "a": 3500.0,
                "mesh_type": "coarse",
                "x": [0.0, 350.0, 700.0, 1050.0, 1400.0, 1750.0, 2100.0, 2450.0, 2800.0, 3150.0, 3500.0],
                "y": [0.0, 350.0, 700.0, 1050.0, 1400.0, 1750.0, 2100.0, 2450.0, 2800.0, 3150.0, 3500.0],
            },
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = process_user_input_with_llm("coarse mesh, a=3500", mock_llm, "", {}, None)
        assert isinstance(result["extracted_multiple"]["x"], list)
        assert len(result["extracted_multiple"]["x"]) == 11

    def test_load_cases_preserved_in_extracted_multiple(self):
        payload = {
            "needs_clarification": False,
            "clarification_question": None,
            "friendly_response": "Tandem setup captured.",
            "use_default": False,
            "use_all_defaults": False,
            "conversation_only": False,
            "understood": {},
            "computed": {
                "load_cases": [
                    {"x1": 0.0, "x2": 300.0, "y1": 1100.0, "y2": 1400.0, "q": 0.667},
                    {"x1": 1200.0, "x2": 1500.0, "y1": 1100.0, "y2": 1400.0, "q": 0.667},
                ],
                "x1": 0.0, "x2": 300.0, "y1": 1100.0, "y2": 1400.0, "q": 0.667,
            },
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = process_user_input_with_llm("tandem axle, two 60kN wheels", mock_llm, "", {}, None)
        assert isinstance(result["extracted_multiple"]["load_cases"], list)
        assert len(result["extracted_multiple"]["load_cases"]) == 2
