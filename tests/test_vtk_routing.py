# tests/test_vtk_routing.py
import json
import pytest
from unittest.mock import MagicMock
from backend.vtk_lookup_agent import (
    _build_vtk_router_system_prompt,
    _parse_agentic_intent,
    LookupIntent,
)


class TestBuildVtkRouterSystemPrompt:
    def test_returns_nonempty_string(self):
        prompt = _build_vtk_router_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    def test_contains_routing_actions(self):
        prompt = _build_vtk_router_system_prompt()
        for action in ["detail", "aggregate", "flexural", "plots", "none"]:
            assert action in prompt

    def test_contains_aggregation_keywords(self):
        prompt = _build_vtk_router_system_prompt()
        assert "max" in prompt
        assert "mean" in prompt
        assert "min" in prompt

    def test_contains_field_mappings(self):
        prompt = _build_vtk_router_system_prompt()
        assert "sxx_top" in prompt or "σ_x" in prompt
        assert "deflection" in prompt or "w" in prompt


class TestParseAgenticIntent:
    def _make_llm_mock(self, response_content: str):
        mock_response = MagicMock()
        mock_response.content = response_content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_aggregate_query_returns_aggregate_intent(self):
        payload = {
            "is_vtk_query": True,
            "action": "aggregate",
            "field": "w",
            "aggregation": "max",
            "component": None,
            "use_abs": False,
            "with_plot": False,
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = _parse_agentic_intent(
            "what is the maximum deflection?",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is not None
        assert result.action == "aggregate"
        assert result.field == "w"
        assert result.aggregation == "max"

    def test_flexural_query_returns_flexural_intent(self):
        payload = {
            "is_vtk_query": True,
            "action": "flexural",
            "field": None,
            "aggregation": None,
            "component": None,
            "use_abs": False,
            "with_plot": False,
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = _parse_agentic_intent(
            "show me the flexural stress report",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is not None
        assert result.action == "flexural"

    def test_non_vtk_query_returns_none(self):
        payload = {
            "is_vtk_query": False,
            "action": "none",
            "field": None,
            "aggregation": None,
            "component": None,
            "use_abs": False,
            "with_plot": False,
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = _parse_agentic_intent(
            "what is the weather today?",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=False,
        )
        assert result is None

    def test_llm_none_returns_none(self):
        result = _parse_agentic_intent(
            "max deflection?",
            llm=None,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is None

    def test_malformed_llm_response_returns_none(self):
        mock_llm = self._make_llm_mock("I cannot classify this.")
        result = _parse_agentic_intent(
            "something about vtk",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is None

    def test_aggregate_with_null_field_routes_to_flexural(self):
        payload = {
            "is_vtk_query": True,
            "action": "aggregate",
            "field": None,
            "aggregation": "max",
            "component": None,
            "use_abs": False,
            "with_plot": False,
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = _parse_agentic_intent(
            "max stress?",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is not None
        assert result.action == "flexural"

    def test_detail_query_returns_detail_intent(self):
        payload = {
            "is_vtk_query": True,
            "action": "detail",
            "field": None,
            "aggregation": None,
            "component": None,
            "use_abs": False,
            "with_plot": False,
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = _parse_agentic_intent(
            "list available vtk fields",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is not None
        assert result.action == "detail"

    def test_plots_query_returns_plots_intent(self):
        payload = {
            "is_vtk_query": True,
            "action": "plots",
            "field": None,
            "aggregation": None,
            "component": None,
            "use_abs": False,
            "with_plot": False,
        }
        mock_llm = self._make_llm_mock(json.dumps(payload))
        result = _parse_agentic_intent(
            "show me the generated plots",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is not None
        assert result.action == "plots"

    def test_llm_exception_returns_none(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("connection error")
        result = _parse_agentic_intent(
            "max deflection?",
            llm=mock_llm,
            default_vtk_file=None,
            allow_implicit=True,
        )
        assert result is None
