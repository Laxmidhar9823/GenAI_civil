# LLM-Only Extraction & VTK Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all deterministic regex/heuristic parsing in `agent_logic.py` and `vtk_lookup_agent.py` with a pure LLM-driven approach using a single expert call with a two-section (`understood`/`computed`) output schema.

**Architecture:** `process_user_input_with_llm` makes one LLM call per turn using a rich domain-knowledge system prompt; the LLM outputs a JSON object with `understood` (what it parsed) and `computed` (all final solver-ready values including node arrays, load patch coords, q, and load_cases). VTK intent routing is cleaned to use only the existing `_parse_agentic_intent` LLM call with no deterministic fallback.

**Tech Stack:** Python 3.12, LangChain (`ChatOllama`, `SystemMessage`, `HumanMessage`), FastAPI, `json`, `math`

---

## File Map

| File | What changes |
|---|---|
| `backend/agent_logic.py` | Delete ~1,400 lines of deterministic helpers; rewrite `create_expert_system_prompt`, `create_extraction_prompt`, `process_user_input_with_llm`; simplify `normalize_mesh_type` / `normalize_load_location` |
| `backend/vtk_lookup_agent.py` | Delete deterministic helper dicts/functions; add `_build_vtk_router_system_prompt`; rewrite `_parse_agentic_intent`; update `handle_vtk_lookup_command` to remove deterministic fallback |
| `backend/main.py` | Remove two `apply_implicit_inferences` calls; replace with `check_physical_feasibility` only; remove `apply_implicit_inferences` import |
| `tests/test_extraction.py` | New — unit tests for prompt builders and result normalization (LLM mocked) |
| `tests/test_vtk_routing.py` | New — unit tests for VTK router (LLM mocked) |

---

## Task 1: Write Failing Tests for Extraction Layer

**Files:**
- Create: `tests/test_extraction.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail (functions not yet rewritten)**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_extraction.py -v 2>&1 | head -40
```

Expected: Multiple failures — tests for `create_expert_system_prompt` will likely pass since the function exists but returns wrong content; tests for result schema will fail since `extracted_multiple` still comes from the old code path.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_extraction.py
git commit -m "test: add failing tests for LLM-only extraction layer"
```

---

## Task 2: Write Failing Tests for VTK Router

**Files:**
- Create: `tests/test_vtk_routing.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_vtk_routing.py
import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_vtk_routing.py -v 2>&1 | head -30
```

Expected: `_build_vtk_router_system_prompt` is not yet defined → ImportError or failures.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_vtk_routing.py
git commit -m "test: add failing tests for LLM-only VTK routing layer"
```

---

## Task 3: Rewrite `create_expert_system_prompt` in `agent_logic.py`

**Files:**
- Modify: `backend/agent_logic.py` (replace the `create_conversational_system_prompt` function and add `create_expert_system_prompt`)

- [ ] **Step 1: Add `import math` to the imports at the top of `agent_logic.py` if not present**

Check line 1 of `backend/agent_logic.py`. If `import math` is not there, add it after `import json`.

- [ ] **Step 2: Replace `create_conversational_system_prompt` with `create_expert_system_prompt`**

Find the function `create_conversational_system_prompt` (around line 2170) and replace the entire function body with:

```python
def create_expert_system_prompt() -> str:
    param_catalog = {
        k: {
            "name": v["name"],
            "unit": v.get("unit", ""),
            "typical_range": v.get("typical_range", ""),
            "simple_explanation": v.get("simple_explanation", ""),
        }
        for k, v in PARAM_INFO.items()
        if not str(k).startswith("_") and k not in ("x", "y")
    }

    return (
        "You are a rigid pavement FEM configuration assistant with expert knowledge of "
        "IS 456, IRC:58, and pavement engineering.\n\n"
        "YOUR ROLE:\n"
        "- Extract parameters from the user's message\n"
        "- Compute ALL derived values yourself using the exact formulas below\n"
        "- Never guess or assume ambiguous inputs — ask exactly ONE clarification question if unclear\n"
        "- Return ONLY a JSON object matching the schema at the end of this prompt\n\n"

        "═══════════════════════════════════════════════════\n"
        "PARAMETER CATALOGUE\n"
        "═══════════════════════════════════════════════════\n"
        + json.dumps(param_catalog, indent=2)
        + "\n\n"

        "═══════════════════════════════════════════════════\n"
        "DISAMBIGUATION RULES — READ CAREFULLY\n"
        "═══════════════════════════════════════════════════\n"
        "1. DIMENSION PAIRS (e.g. '5000×2000', '350×300'):\n"
        "   A pair WITHOUT the words 'contact', 'tyre', 'tire', 'patch', or 'over' is AMBIGUOUS.\n"
        "   NEVER assume it is slab dimensions OR contact area without clear context.\n"
        "   If ambiguous → needs_clarification=true, ask: 'Is [N]×[M] mm the slab size or the tyre contact dimensions?'\n"
        "   CLEAR: 'slab 5m×3m' → a=5000, b=3000\n"
        "   CLEAR: 'contact area 350×300' → contact lx=350, ly=300\n"
        "   CLEAR: 'over 350×300 mm' or 'tyre contact 350×300' → contact area\n"
        "   AMBIGUOUS: '5000×2000' alone → ask\n\n"
        "2. LOAD WITHOUT CONTACT AREA:\n"
        "   If load (kN) is given but no contact dimensions → needs_clarification=true.\n"
        "   Ask: 'What are the tyre contact dimensions (length × width in mm)?'\n\n"
        "3. EDGE SPECIFICATION:\n"
        "   If user says 'edge load' without specifying which edge → needs_clarification=true.\n"
        "   Ask: 'Which edge — x=0 (near), x=a (far), y=0 (left), or y=b (right)?'\n"
        "   Exception: 'corner edge' implies corner; directional hints ('along the length') imply y=0/y=b.\n\n"

        "═══════════════════════════════════════════════════\n"
        "MANDATORY COMPUTATION FORMULAS\n"
        "═══════════════════════════════════════════════════\n"
        "1. ELASTIC MODULUS FROM GRADE (IS 456):\n"
        "   E (MPa) = 5000 × √fck   [fck = characteristic compressive strength in MPa]\n"
        "   M20→22361, M25→25000, M30→27386, M35→29580, M40→31623 MPa\n\n"
        "2. CONTACT PRESSURE q:\n"
        "   q (MPa) = P_kN × 1000 / (Lx_mm × Ly_mm)\n"
        "   Example: 80 kN, 350×300 mm → q = 80000 / 105000 = 0.7619 MPa\n\n"
        "3. POISSON RATIO DEFAULT (IRC:58): ν = 0.15 — apply silently, do not ask.\n\n"
        "4. SUBGRADE MODULUS K: one K value → Kx=Ky=Kz=K (unless user differentiates).\n\n"
        "5. NODE COORDINATE ARRAYS (always compute both x[] and y[]):\n"
        "   coarse → 11 nodes: x[i] = i × (a/10)  for i=0..10\n"
        "   medium → 16 nodes: x[i] = i × (a/15)  for i=0..15\n"
        "   fine   → 31 nodes: x[i] = i × (a/30)  for i=0..30\n"
        "   Round each value to 6 decimal places.\n"
        "   Example: medium, a=5000 → x=[0, 333.333333, 666.666667, ..., 5000.0]\n\n"
        "6. LOAD PATCH COORDINATES (single wheel, contact Lx × Ly mm):\n"
        "   corner:    x1=0,        x2=Lx,       y1=0,          y2=Ly\n"
        "   edge x=0:  x1=0,        x2=Lx,       y1=(b-Ly)/2,   y2=(b+Ly)/2\n"
        "   edge x=a:  x1=a-Lx,     x2=a,        y1=(b-Ly)/2,   y2=(b+Ly)/2\n"
        "   edge y=0:  x1=(a-Lx)/2, x2=(a+Lx)/2, y1=0,          y2=Ly\n"
        "   edge y=b:  x1=(a-Lx)/2, x2=(a+Lx)/2, y1=b-Ly,       y2=b\n"
        "   interior:  x1=(a-Lx)/2, x2=(a+Lx)/2, y1=(b-Ly)/2,   y2=(b+Ly)/2\n\n"
        "7. MULTI-WHEEL load_cases (each entry = one tyre patch {x1,x2,y1,y2,q}):\n"
        "   DUAL TYRES (side by side, across slab y-direction, wheel spacing s):\n"
        "     Wheel 1: y_center = b/2 - s/2 → y1=y_center-Ly/2, y2=y_center+Ly/2\n"
        "     Wheel 2: y_center = b/2 + s/2 → y1=y_center-Ly/2, y2=y_center+Ly/2\n"
        "     Both wheels: x1=(a-Lx)/2, x2=(a+Lx)/2\n"
        "   TANDEM AXLE (two axles along slab x-direction, axle spacing d):\n"
        "     Axle 1: x_center = a/2 - d/2 → x1=x_center-Lx/2, x2=x_center+Lx/2\n"
        "     Axle 2: x_center = a/2 + d/2 → x1=x_center-Lx/2, x2=x_center+Lx/2\n"
        "     Both axles: y1=(b-Ly)/2, y2=(b+Ly)/2\n"
        "   TANDEM + DUAL: 4 patches (2 axles × 2 tyre positions)\n"
        "   PER-WHEEL q: if axle total load P_total with N wheels → q = (P_total/N)×1000/(Lx×Ly)\n"
        "   Single wheel: load_cases=[], put x1/x2/y1/y2 directly in computed.\n"
        "   Multi-wheel: populate load_cases; ALSO copy first entry values to top-level x1/x2/y1/y2/q.\n\n"
        "8. UNIT CONVERSIONS:\n"
        "   Length: 1m=1000mm, 1cm=10mm, 1ft=304.8mm, 1inch=25.4mm\n"
        "   Load:   1kN=1000N, 1ton=9.81kN, 1tonne=9.81kN, 1kg=0.00981kN\n"
        "   Pressure: 1kPa=0.001MPa, 1psi=0.006895MPa, 1GPa=1000MPa\n\n"

        "═══════════════════════════════════════════════════\n"
        "WHEN TO ASK FOR CLARIFICATION\n"
        "═══════════════════════════════════════════════════\n"
        "ASK (needs_clarification=true) when:\n"
        "- Dimension pair given without 'contact'/'tyre'/'patch' context\n"
        "- Load given but no contact dimensions\n"
        "- Load location not specified or cannot be inferred\n"
        "- 'edge' without specifying which edge\n"
        "- Contradictory values (contact area > slab)\n\n"
        "DO NOT ASK when:\n"
        "- E is computable from grade given\n"
        "- q is computable from load + contact dims given\n"
        "- ν not stated → use 0.15 silently\n"
        "- mesh_type known → compute node arrays without asking\n"
        "- load_location + contact dims known → compute patch coords without asking\n\n"

        "═══════════════════════════════════════════════════\n"
        "RECOGNISING SPECIAL INTENTS\n"
        "═══════════════════════════════════════════════════\n"
        "'yes'/'ok'/'sure'/'that\\'s fine' when asked about a param → use_default=true\n"
        "'use all defaults'/'fill with defaults' → use_all_defaults=true, compute known derived values\n"
        "Greetings ('hi','hello') → conversation_only=true, warm greeting, no params extracted\n"
        "Help requests ('help','what can you do') → conversation_only=true, explain capabilities\n"
        "'what is [param]'/'explain [param]' → conversation_only=true, explain the parameter\n\n"

        "═══════════════════════════════════════════════════\n"
        "OUTPUT JSON SCHEMA — return ONLY this object, no other text\n"
        "═══════════════════════════════════════════════════\n"
        "{\n"
        '  "needs_clarification": false,\n'
        '  "clarification_question": null,\n'
        '  "friendly_response": "Brief warm confirmation of what was understood and computed.",\n'
        '  "use_default": false,\n'
        '  "use_all_defaults": false,\n'
        '  "conversation_only": false,\n'
        '  "understood": {\n'
        '    // Human-readable summary of what the user provided\n'
        '    // e.g. "slab_a_mm": 5000, "fck_mpa": 30, "load_kn": 80\n'
        '    // Omit keys the user did not mention\n'
        '  },\n'
        '  "computed": {\n'
        '    // Solver-ready final values — use EXACT key names shown\n'
        '    // Only include params that were provided or derivable from provided values\n'
        '    "a": <number|omit>,\n'
        '    "b": <number|omit>,\n'
        '    "t": <number|omit>,\n'
        '    "Emod": <number|omit>,\n'
        '    "nu": <number|omit>,\n'
        '    "Kx": <number|omit>,\n'
        '    "Ky": <number|omit>,\n'
        '    "Kz": <number|omit>,\n'
        '    "q": <number|omit>,\n'
        '    "x1": <number|omit>,\n'
        '    "x2": <number|omit>,\n'
        '    "y1": <number|omit>,\n'
        '    "y2": <number|omit>,\n'
        '    "mesh_type": <"coarse"|"medium"|"fine"|omit>,\n'
        '    "load_location": <"corner"|"edge"|"interior"|omit>,\n'
        '    "x": [<evenly spaced nodes from 0 to a>|omit],\n'
        '    "y": [<evenly spaced nodes from 0 to b>|omit],\n'
        '    "load_cases": []  // empty for single wheel; [{x1,x2,y1,y2,q},...] for multi-wheel\n'
        '  }\n'
        "}\n\n"
        "WHEN needs_clarification=true: set understood={}, computed={}, populate clarification_question.\n"
        "NEVER output anything outside the JSON object."
    )
```

- [ ] **Step 3: Run the prompt tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_extraction.py::TestCreateExpertSystemPrompt -v
```

Expected: All 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/agent_logic.py
git commit -m "feat: add create_expert_system_prompt with full rigid pavement domain knowledge"
```

---

## Task 4: Rewrite `create_extraction_prompt` in `agent_logic.py`

**Files:**
- Modify: `backend/agent_logic.py` (replace existing `create_extraction_prompt` function, around line 2215)

- [ ] **Step 1: Replace `create_extraction_prompt`**

Find the existing `create_extraction_prompt` function (around line 2215) and replace its entire body:

```python
def create_extraction_prompt(
    user_input: str,
    context: str,
    current_asking: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    if current_asking and current_asking in PARAM_INFO:
        info = PARAM_INFO[current_asking]
        default_val = get_default_value(current_asking)
        asking_block = (
            f"CURRENTLY ASKING ABOUT: {info['name']} ({current_asking})\n"
            f"  Unit: {info.get('unit', 'N/A')}\n"
            f"  Typical range: {info.get('typical_range', 'N/A')}\n"
            f"  Safe default: {default_val}\n"
            f"  If user says yes/ok/sure/that's fine → set use_default=true\n"
        )
    else:
        asking_block = (
            "CURRENTLY ASKING ABOUT: General input\n"
            "  Extract any and all parameters mentioned in the message.\n"
        )

    collected_params = {
        k: v for k, v in (params or {}).items()
        if not str(k).startswith("_")
    }
    params_block = (
        f"ALREADY COLLECTED PARAMETERS:\n{json.dumps(collected_params, indent=2)}\n"
        if collected_params
        else "ALREADY COLLECTED PARAMETERS: (none yet)\n"
    )

    return (
        f"### CONVERSATION CONTEXT\n{context}\n\n"
        f"### {asking_block}\n"
        f"### {params_block}\n"
        f"### USER MESSAGE\n\"{user_input}\"\n\n"
        "### TASK\n"
        "Using the system prompt rules, extract and compute all relevant parameters "
        "from the user message. Return ONLY the JSON object. No markdown fences, "
        "no explanation, no prose before or after.\n\n"
        "JSON:"
    )
```

- [ ] **Step 2: Run the prompt tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_extraction.py::TestCreateExtractionPrompt -v
```

Expected: All 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/agent_logic.py
git commit -m "feat: rewrite create_extraction_prompt for two-section LLM schema"
```

---

## Task 5: Rewrite `process_user_input_with_llm` in `agent_logic.py`

**Files:**
- Modify: `backend/agent_logic.py` (replace existing `process_user_input_with_llm`, around line 2344)

- [ ] **Step 1: Replace `process_user_input_with_llm`**

Find the existing `process_user_input_with_llm` function (around line 2344) and replace its entire body. The function signature stays the same:

```python
def process_user_input_with_llm(
    user_input: str,
    llm: ChatOllama,
    context: str,
    params: Optional[Dict[str, Any]] = None,
    current_asking: Optional[str] = None,
) -> Dict[str, Any]:
    _FALLBACK: Dict[str, Any] = {
        "needs_clarification": True,
        "clarification_question": None,
        "friendly_response": (
            "I didn't quite catch that. Could you rephrase? "
            "You can share values like 'slab 5m × 3m, 250mm thick, M30, K=60'."
        ),
        "use_default": False,
        "use_all_defaults": False,
        "conversation_only": False,
        "understood": {},
        "computed": {},
        "extracted_multiple": {},
        "extracted_units": {},
        "parameter_key": current_asking,
        "understood_value": None,
        "original_unit": None,
    }

    def _parse_json(content: str) -> Optional[Dict]:
        body = (content or "").strip()
        if body.startswith("```"):
            first_nl = body.find("\n")
            if first_nl != -1:
                body = body[first_nl + 1:]
            if body.endswith("```"):
                body = body[:-3]
            body = body.strip()
        start = body.find("{")
        end = body.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(body[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _normalize_result(raw: Dict) -> Dict:
        raw.setdefault("needs_clarification", False)
        raw.setdefault("clarification_question", None)
        raw.setdefault("friendly_response", "")
        raw.setdefault("use_default", False)
        raw.setdefault("use_all_defaults", False)
        raw.setdefault("conversation_only", False)
        raw.setdefault("understood", {})
        raw.setdefault("computed", {})

        computed = raw.get("computed") or {}
        raw["extracted_multiple"] = {
            k: v for k, v in computed.items() if not str(k).startswith("_")
        }
        raw.setdefault("extracted_units", {})
        raw.setdefault("parameter_key", current_asking)
        raw.setdefault("understood_value", None)
        raw.setdefault("original_unit", None)
        return raw

    system_msg = SystemMessage(content=create_expert_system_prompt())
    human_content = create_extraction_prompt(user_input, context, current_asking, params)

    try:
        response = _invoke_llm_with_timeout(llm, [system_msg, HumanMessage(content=human_content)])
        content = str(getattr(response, "content", "") or "")
        result = _parse_json(content)

        if result is None:
            retry_content = human_content + "\n\nReturn ONLY the JSON object. No markdown, no explanation."
            response2 = _invoke_llm_with_timeout(llm, [system_msg, HumanMessage(content=retry_content)])
            content2 = str(getattr(response2, "content", "") or "")
            result = _parse_json(content2)

        if result is None:
            return _FALLBACK

        return _normalize_result(result)

    except TimeoutError:
        fallback = dict(_FALLBACK)
        fallback["friendly_response"] = (
            "I'm having trouble reaching the model right now. "
            "If you share the key numbers (slab size, thickness, concrete grade, "
            "K value, wheel load/contact area), I can still capture them — "
            "or say 'let's begin' for guided setup."
        )
        return fallback

    except Exception as exc:
        err_text = str(exc).strip().lower()
        if "model" in err_text and "not found" in err_text:
            msg = (
                "The selected Ollama model is not installed. "
                "Please choose an available model in settings."
            )
        elif "unauthorized" in err_text or "401" in err_text:
            msg = (
                "Authentication failed (401). "
                "Please check your API key in settings or switch to a local model."
            )
        elif "connection" in err_text or "cannot reach" in err_text:
            msg = "Cannot reach the Ollama server. Please ensure Ollama is running and try again."
        else:
            msg = (
                "I hit an internal processing issue. "
                "Please try again with key values (e.g., length 5m, width 5m, thickness 200mm)."
            )
        fallback = dict(_FALLBACK)
        fallback["friendly_response"] = msg
        return fallback
```

- [ ] **Step 2: Run the process_user_input tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_extraction.py::TestProcessUserInputWithLLM -v
```

Expected: All 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/agent_logic.py
git commit -m "feat: rewrite process_user_input_with_llm — single expert LLM call, two-section schema"
```

---

## Task 6: Simplify `normalize_mesh_type` and `normalize_load_location`

**Files:**
- Modify: `backend/agent_logic.py` (around lines 643 and 656)

These are still called by `main.py` as safety guards on the LLM's output. Simplify them to canonical-value validators only — no synonym expansion needed since the LLM now outputs canonical values.

- [ ] **Step 1: Replace both normalization functions**

Find `normalize_mesh_type` (around line 643) and `normalize_load_location` (around line 656) and replace both:

```python
def normalize_mesh_type(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"coarse", "1"}:
        return "coarse"
    if text in {"medium", "2"}:
        return "medium"
    if text in {"fine", "3"}:
        return "fine"
    return None


def normalize_load_location(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if text in {"corner", "1"}:
        return "corner"
    if text in {"edge", "2"}:
        return "edge"
    if text in {"interior", "3"}:
        return "interior"
    return None
```

- [ ] **Step 2: Verify the server still imports without error**

```bash
cd /home/surriya_gokul/GenAI_civil && python -c "from backend.agent_logic import normalize_mesh_type, normalize_load_location; print(normalize_mesh_type('medium'), normalize_load_location('corner'))"
```

Expected output: `medium corner`

- [ ] **Step 3: Commit**

```bash
git add backend/agent_logic.py
git commit -m "refactor: simplify normalize_mesh_type and normalize_load_location to canonical validators"
```

---

## Task 7: Delete Deterministic Code from `agent_logic.py`

**Files:**
- Modify: `backend/agent_logic.py`

Delete all of these functions and constants in one pass. They are no longer called by any remaining code.

- [ ] **Step 1: Verify nothing in the remaining codebase imports the functions to be deleted**

```bash
grep -n "_extract_engineering_candidates\|parse_multi_param_candidates\|infer_semantic_choices\|handle_conversational_intent\|apply_implicit_inferences\|infer_load_patch\|infer_nodes_from_mesh\|_extract_first_number\|_tokenize_for_numbers\|_parse_node_list\|_build_group_wheel\|_compute_patch_from_desc\|SINGLE_DEFAULT_PATTERNS\|PARAM_ALIASES\|MESH_LEVEL_ELEMENTS" \
  /home/surriya_gokul/GenAI_civil/backend/main.py \
  /home/surriya_gokul/GenAI_civil/backend/vtk_lookup_agent.py \
  /home/surriya_gokul/GenAI_civil/app.py 2>/dev/null
```

Expected: Only `apply_implicit_inferences` appears in `main.py` (which gets removed in Task 9). Everything else should show zero hits.

- [ ] **Step 2: Delete the following functions and constants from `agent_logic.py`**

Delete these functions entirely (find each `def <name>` and delete from that line to the next `def` or blank-line boundary):

**Constants to delete (find by name and delete the entire block):**
- `SINGLE_DEFAULT_PATTERNS` (around line 606) — the dict `{ "skip", "ok", ... }`
- `PARAM_ALIASES` (around line 614) — the full dict
- `MESH_LEVEL_ELEMENTS` (around line 249) — the `{"coarse": 10, ...}` dict
- `STANDARD_POISSON_RATIO` (around line 255) — the float assignment
- `_GREETINGS`, `_THANKS`, `_GOODBYES`, `_HELP_PATTERNS`, `_EXPLAIN_PATTERNS` (around lines 258–305) — all five sets

**Functions to delete:**
- `_normalize_text` (around line 308)
- `_contains_default_keyword` (around line 312)
- `_looks_like_greeting` (around line 319)
- `_looks_like_thanks` (around line 330)
- `_looks_like_goodbye` (around line 335)
- `_looks_like_help_request` (around line 340)
- `_looks_like_explanation_request` (around line 345)
- `handle_conversational_intent` (around line 430)
- `infer_semantic_choices` (around line 855)
- `_linspace_nodes` (around line 898)
- `infer_nodes_from_mesh` (around line 907)
- `infer_load_patch_from_location` (around line 913)
- `infer_load_patch_from_contact_area` (around line 954)
- `_extract_spacing_mm` (around line 972)
- `_extract_wheel_spacing_mm` (around line 985)
- `_extract_axle_spacing_mm` (around line 997)
- `_offset_load_patch` (around line 1009)
- `_detect_number_of_wheels` (around line 1031)
- `_extract_edge_distances_mm` (around line 1040)
- `_center_to_patch` (around line 1052)
- `_nearest_mesh_point` (around line 1063)
- `_compute_patch_from_description` (around line 1069)
- `_build_group_wheel_load_cases` (around line 1122)
- `apply_implicit_inferences` (around line 1214)
- `_clamp` (around line 675)
- `_resolve_reference_location_token` (around line 681)
- `_reference_point_from_token` (around line 742)
- `_first_wheel_center_from_token` (around line 762)
- `_dual_spacing_direction` (around line 791)
- `_clamp_center_to_slab` (around line 809)
- `_center_to_bounded_patch` (around line 837)
- `_compact_value` (around line 1492) — check it is not used elsewhere first
- `_extract_first_number_with_unit` (around line 1500)
- `_tokenize_for_numbers` (around line 1543)
- `_parse_number_token_with_unit` (around line 1582)
- `_extract_all_numbers_with_units` (around line 1609)
- `_parse_node_list_candidates` (around line 1627)
- `_find_phrase_indices` (around line 1648)
- `_length_to_mm` (around line 1659)
- `_compute_contact_pressure_q_mpa` (around line 1680)
- `_extract_assumed_values` (around line 1713)
- `_extract_engineering_candidates` (around line 1758)
- `parse_multi_param_candidates` (around line 2050)
- `_detect_extraction_ambiguity` (around line 2314)

Also delete the `import re` line at the top if `re` is no longer used anywhere in the file. Check first:

```bash
grep -n "re\." /home/surriya_gokul/GenAI_civil/backend/agent_logic.py | grep -v "friendly_response\|required\|served\|user_provided\|ensure\|measure\|feature\|procedure\|stored" | head -20
```

- [ ] **Step 3: Verify the module imports cleanly**

```bash
cd /home/surriya_gokul/GenAI_civil && python -c "
from backend.agent_logic import (
    process_user_input_with_llm, create_expert_system_prompt,
    create_extraction_prompt, PARAM_INFO, PARAM_ORDER, DEFAULT_VALUES,
    get_default_value, build_conversation_context, validate_single_param,
    check_physical_feasibility, generate_autonomous_assistant_message,
    build_final_configuration, normalize_mesh_type, normalize_load_location,
)
print('OK')
"
```

Expected: `OK` with no errors.

- [ ] **Step 4: Run all extraction tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_extraction.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/agent_logic.py
git commit -m "refactor: delete ~1400 lines of deterministic parsing from agent_logic.py"
```

---

## Task 8: Add `_build_vtk_router_system_prompt` and Rewrite `_parse_agentic_intent` in `vtk_lookup_agent.py`

**Files:**
- Modify: `backend/vtk_lookup_agent.py`

- [ ] **Step 1: Add `_build_vtk_router_system_prompt` function**

Add this function right before `_parse_agentic_intent` (around line 427):

```python
def _build_vtk_router_system_prompt() -> str:
    return (
        "You are a VTK result query router for a rigid pavement FEM analysis tool.\n"
        "Classify the user's query and return STRICT JSON only — no prose, no explanation.\n\n"
        "OUTPUT SCHEMA:\n"
        "{\n"
        '  "is_vtk_query": boolean,\n'
        '  "action": "detail" | "aggregate" | "flexural" | "plots" | "none",\n'
        '  "field": string | null,\n'
        '  "aggregation": "mean"|"max"|"min"|"sum"|"std"|"var"|"median"|"count"|"p95"|"p05"|null,\n'
        '  "component": "x"|"y"|"z"|"magnitude"|null,\n'
        '  "use_abs": boolean,\n'
        '  "with_plot": boolean\n'
        "}\n\n"
        "ROUTING RULES:\n"
        "- detail: user asks what fields exist, wants summary/info/list of available fields.\n"
        "- aggregate: user asks for a numeric statistic over one specific result field.\n"
        "- flexural: user asks about bending/flexural stress, top/bottom surface stress, σ_x, σ_y.\n"
        "- plots: user asks to see/show/view generated contour plots or images.\n"
        "- none: query is unrelated to FEM/VTK results.\n\n"
        "FIELD MAPPING (use exact canonical names):\n"
        "- deflection / vertical displacement / out-of-plane displacement / w → field='w'\n"
        "- x displacement / longitudinal displacement / u → field='u'\n"
        "- y displacement / transverse displacement / v → field='v'\n"
        "- rotation about y / slope x / theta_x → field='theta_x'\n"
        "- rotation about x / slope y / theta_y → field='theta_y'\n"
        "- σ_x top / sxx top / top normal stress x → field='sxx_top'\n"
        "- σ_x bottom / sxx bottom → field='sxx_bottom'\n"
        "- σ_y top / syy top → field='syy_top'\n"
        "- σ_y bottom / syy bottom → field='syy_bottom'\n"
        "- shear top / τ_xy top / sxy top → field='sxy_top'\n"
        "- shear bottom / sxy bottom → field='sxy_bottom'\n"
        "- If user asks about 'stress' generally → route to flexural (action='flexural', field=null)\n"
        "- If field is unclear for aggregate → set field=null (will route to flexural)\n\n"
        "AGGREGATION MAPPING:\n"
        "- maximum / highest / peak / max → 'max'\n"
        "- minimum / lowest / min → 'min'\n"
        "- average / mean / avg → 'mean'\n"
        "- standard deviation / std / stdev → 'std'\n"
        "- total / sum → 'sum'\n"
        "- median → 'median'\n"
        "- 95th percentile / p95 → 'p95'\n"
        "- 5th percentile / p05 → 'p05'\n\n"
        "allow_implicit=true means treat implicit result references (deflection, stress) "
        "as VTK queries even without the word 'vtk'.\n"
    )
```

- [ ] **Step 2: Replace `_parse_agentic_intent` entirely**

Find the existing `_parse_agentic_intent` function (around line 427) and replace its entire body:

```python
def _parse_agentic_intent(
    user_input: str,
    *,
    llm: Optional[ChatOllama],
    default_vtk_file: Optional[Path],
    allow_implicit: bool,
) -> Optional[LookupIntent]:
    if llm is None:
        return None

    vtk_file = _extract_vtk_path(user_input, default_vtk_file=default_vtk_file)

    available_fields: List[str] = []
    if vtk_file.exists():
        try:
            info = describe_dataset(str(vtk_file))
            available_fields = list(info.get("available_fields", {}).get("point_data", []) or [])
        except Exception:
            pass
    if not available_fields:
        available_fields = [
            "w", "u", "v", "theta_x", "theta_y",
            "sxx_top", "sxx_bottom", "syy_top", "syy_bottom",
            "sxy_top", "sxy_bottom", "sxx_membrane", "syy_membrane",
            "sxy_membrane", "displacement",
        ]

    system = SystemMessage(content=_build_vtk_router_system_prompt())
    human = HumanMessage(content=_build_planner_prompt(
        user_input=user_input,
        vtk_file=vtk_file,
        allow_implicit=allow_implicit,
    ))

    try:
        response = _invoke_llm_with_timeout(llm, [system, human], timeout_seconds=10.0)
        content = str(getattr(response, "content", "") or "")
        payload = _safe_parse_json_object(content)
    except Exception:
        return None

    if not payload or not bool(payload.get("is_vtk_query")):
        return None

    action = _normalize_text(str(payload.get("action") or "none"))
    if action in {"", "none", "null"}:
        return None

    if action == "plots":
        return LookupIntent(action="plots", vtk_file=vtk_file)

    if action == "detail":
        return LookupIntent(action="detail", vtk_file=vtk_file)

    if action == "flexural":
        with_plot = bool(payload.get("with_plot"))
        return LookupIntent(action="flexural", vtk_file=vtk_file, with_plot=with_plot)

    if action == "aggregate":
        field = payload.get("field")
        if not field:
            with_plot = bool(payload.get("with_plot"))
            return LookupIntent(action="flexural", vtk_file=vtk_file, with_plot=with_plot)

        aggregation = str(payload.get("aggregation") or "mean")
        component_raw = payload.get("component")
        component = (
            str(component_raw)
            if isinstance(component_raw, str) and component_raw in {"x", "y", "z", "magnitude"}
            else None
        )

        return LookupIntent(
            action="aggregate",
            vtk_file=vtk_file,
            aggregation=aggregation,
            field=str(field),
            component=component,
            use_abs=bool(payload.get("use_abs")),
            field_explicit=True,
        )

    return None
```

- [ ] **Step 3: Update `_build_planner_prompt` to accept an optional `available_fields` param**

Find `_build_planner_prompt` (around line 381) and update its signature and body to match what `_parse_agentic_intent` now calls:

```python
def _build_planner_prompt(
    *,
    user_input: str,
    vtk_file: Path,
    allow_implicit: bool,
) -> str:
    available_fields: List[str] = []
    if vtk_file.exists():
        try:
            info = describe_dataset(str(vtk_file))
            available_fields = list(info.get("available_fields", {}).get("point_data", []) or [])
        except Exception:
            available_fields = []

    if not available_fields:
        available_fields = sorted([
            "w", "u", "v", "theta_x", "theta_y",
            "sxx_top", "sxx_bottom", "syy_top", "syy_bottom",
            "sxy_top", "sxy_bottom", "sxx_membrane", "syy_membrane",
            "sxy_membrane", "displacement",
        ])

    return (
        f"allow_implicit={str(bool(allow_implicit)).lower()}\n"
        f"vtk_file={vtk_file}\n"
        f"AVAILABLE_FIELDS={', '.join(available_fields)}\n\n"
        f"USER_QUERY: {user_input}\n\n"
        "Return the JSON object."
    )
```

- [ ] **Step 4: Run the VTK routing tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_vtk_routing.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/vtk_lookup_agent.py
git commit -m "feat: add VTK router system prompt and rewrite _parse_agentic_intent to pure LLM routing"
```

---

## Task 9: Delete Deterministic Helpers from `vtk_lookup_agent.py`

**Files:**
- Modify: `backend/vtk_lookup_agent.py`

- [ ] **Step 1: Remove deterministic `_parse_intent` fallback from `handle_vtk_lookup_command`**

Find `handle_vtk_lookup_command` (around line 862). Change this block:

```python
# BEFORE — find and replace this entire block:
intent: Optional[LookupIntent] = None
if prefer_agentic and llm is not None:
    intent = _parse_agentic_intent(
        user_input,
        llm=llm,
        default_vtk_file=fallback_path,
        allow_implicit=allow_implicit,
    )

if intent is None:
    intent = _parse_intent(
        user_input,
        default_vtk_file=fallback_path,
        allow_implicit=allow_implicit,
    )
```

Replace with:

```python
# AFTER — pure agentic path only:
intent: Optional[LookupIntent] = _parse_agentic_intent(
    user_input,
    llm=llm,
    default_vtk_file=fallback_path,
    allow_implicit=allow_implicit,
)
```

Also remove the `prefer_agentic` parameter from the function signature (it is no longer needed). Update the function signature from:

```python
def handle_vtk_lookup_command(
    user_input: str,
    *,
    default_vtk_file: Optional[str] = None,
    allow_implicit: bool = False,
    api_base_url: str = "",
    llm: Optional[ChatOllama] = None,
    prefer_agentic: bool = False,
    available_plot_urls: Optional[List[str]] = None,
    narration_url: Optional[str] = None,
    narration_model: Optional[str] = None,
    narration_api_key: Optional[str] = None,
) -> Optional[str]:
```

to:

```python
def handle_vtk_lookup_command(
    user_input: str,
    *,
    default_vtk_file: Optional[str] = None,
    allow_implicit: bool = False,
    api_base_url: str = "",
    llm: Optional[ChatOllama] = None,
    available_plot_urls: Optional[List[str]] = None,
    narration_url: Optional[str] = None,
    narration_model: Optional[str] = None,
    narration_api_key: Optional[str] = None,
) -> Optional[str]:
```

Also remove the `plot_urls` line from the function body:

```python
# Remove this line (no longer needed — plot_urls is still available_plot_urls param):
plot_urls = [url for url in (available_plot_urls or []) if isinstance(url, str) and url.strip()]
```

Wait — `plot_urls` is used later in the function. Keep it as-is, just remove `prefer_agentic`.

- [ ] **Step 2: Delete deterministic helper functions and dicts**

Delete the following from `vtk_lookup_agent.py` (find each and delete the entire block):

**Module-level constants to delete:**
- `_FIELD_ALIASES` dict (around line 39)
- `_AGG_KEYWORDS` dict (around line 57)
- `_DETAIL_HINTS` tuple (around line 86)
- `_FLEXURAL_HINTS` tuple (around line 97)
- `_IMPLICIT_RESULT_HINTS` tuple (around line 110)
- `_AGG_CANONICAL` set (around line 124)
- `_PLOT_REQUEST_HINTS` tuple (around line 126)
- `_GENERIC_STRESS_TERMS` frozenset (around line 139)
- `_FLEXURAL_STRESS_FIELDS` list (around line 142)
- `_QUALITATIVE_ANALYSIS_TERMS` frozenset (around line 146)
- `_FIELD_LISTING_TERMS` frozenset (around line 154)

**Functions to delete:**
- `_normalize_text` (around line 160)
- `_default_aggregate_field_for_query` (around line 164)
- `_detect_aggregation` (around line 237)
- `_detect_component` (around line 245)
- `_detect_field` (around line 264)
- `_coerce_aggregation` (around line 287)
- `_is_plot_request` (around line 298)
- `_is_lookup_intent` (around line 303)
- `_parse_intent` (around line 339)

Also check and remove `import re` from `vtk_lookup_agent.py` if re is no longer used:

```bash
grep -n "re\." /home/surriya_gokul/GenAI_civil/backend/vtk_lookup_agent.py | head -10
```

- [ ] **Step 3: Verify `vtk_lookup_agent.py` imports cleanly**

```bash
cd /home/surriya_gokul/GenAI_civil && python -c "
from backend.vtk_lookup_agent import handle_vtk_lookup_command, build_vtk_stats_summary
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Run all VTK tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_vtk_routing.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/vtk_lookup_agent.py
git commit -m "refactor: delete ~300 lines of deterministic VTK routing helpers"
```

---

## Task 10: Update `main.py` — Remove `apply_implicit_inferences`, Fix `prefer_agentic`

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Remove `apply_implicit_inferences` from the import block**

Find the import block (around line 15) and remove `apply_implicit_inferences` from the import list.

Also remove `normalize_load_location` and `normalize_mesh_type` if you want to keep them as internal-only to `agent_logic.py`. However, since `main.py` still calls them (lines ~665, ~667, ~791, ~793) as safety guards, **keep them in the import**. They now just do lightweight canonical validation.

Remove only `apply_implicit_inferences` and `infer_semantic_choices` from the import list.

- [ ] **Step 2: Fix the `handle_vtk_lookup_command` call — remove `prefer_agentic`**

Find the call around line 515:

```python
# BEFORE:
lookup_response = handle_vtk_lookup_command(
    req.user_input,
    default_vtk_file=analysis_vtk or None,
    allow_implicit=bool(analysis_vtk),
    api_base_url=API_BASE_URL,
    llm=vtk_llm,
    prefer_agentic=True,
    available_plot_urls=state.analysis_plot_files,
    narration_url=req.llm_config.ollama_url,
    narration_model=VTK_AGENT_MODEL,
    narration_api_key=api_key,
)
```

Replace with:

```python
# AFTER:
lookup_response = handle_vtk_lookup_command(
    req.user_input,
    default_vtk_file=analysis_vtk or None,
    allow_implicit=bool(analysis_vtk),
    api_base_url=API_BASE_URL,
    llm=vtk_llm,
    available_plot_urls=state.analysis_plot_files,
    narration_url=req.llm_config.ollama_url,
    narration_model=VTK_AGENT_MODEL,
    narration_api_key=api_key,
)
```

- [ ] **Step 3: Remove the `apply_implicit_inferences` call in the `use_all_defaults` path**

Find around line 684:

```python
# BEFORE (in the use_all_defaults block):
_, inference_notes = apply_implicit_inferences(state.params, state.user_provided_keys)
inference_notes.extend(check_physical_feasibility(state.params))
```

Replace with:

```python
# AFTER:
inference_notes = list(check_physical_feasibility(state.params))
```

- [ ] **Step 4: Remove the `apply_implicit_inferences` call in the normal extraction path**

Find around line 816:

```python
# BEFORE (in the normal extraction block):
inferred_keys, inference_notes = apply_implicit_inferences(state.params, state.user_provided_keys)
inference_notes.extend(check_physical_feasibility(state.params))
for inferred_key in inferred_keys:
    if inferred_key in tentative:
        tentative[inferred_key] = state.params[inferred_key]
applied.extend([k for k in inferred_keys if k not in applied])
```

Replace with:

```python
# AFTER:
inference_notes = list(check_physical_feasibility(state.params))
```

Also remove the comment on line ~810 that says:
```python
# so apply_implicit_inferences() doesn't overwrite them with the 20% heuristic.
```
since it is no longer accurate.

- [ ] **Step 5: Verify the server starts without errors**

```bash
cd /home/surriya_gokul/GenAI_civil && python -c "
from backend.main import app
print('FastAPI app loaded OK')
"
```

Expected: `FastAPI app loaded OK`

- [ ] **Step 6: Run all tests**

```bash
cd /home/surriya_gokul/GenAI_civil && python -m pytest tests/test_extraction.py tests/test_vtk_routing.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add backend/main.py
git commit -m "refactor: remove apply_implicit_inferences from main.py, remove prefer_agentic flag"
```

---

## Task 11: Smoke Test the Full Stack

**Files:**
- No file changes — verification only

- [ ] **Step 1: Start the backend server**

```bash
cd /home/surriya_gokul/GenAI_civil && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Keep this running in one terminal.

- [ ] **Step 2: Send a contact-area-ambiguity test message (the original bug)**

In a second terminal:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "contact area 5000x2000",
    "state": {"mode": "free", "params": {}, "messages": [], "user_provided_keys": [], "current_asking": null, "memory": null, "analysis_generated": false, "analysis_vtk_file": null, "analysis_plot_files": []},
    "llm_config": {"ollama_url": "http://localhost:11434", "model": "llama3.2", "api_key": null}
  }' | python -m json.tool | grep -A3 "assistant_message\|needs_clarification"
```

Expected: The assistant asks which edge OR correctly identifies it as contact area and asks for the load — it must NOT silently assign a=5000, b=2000.

- [ ] **Step 3: Send a complete one-shot message**

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "M30 concrete slab 5m x 3m, 250mm thick, K=60, 80kN wheel load at corner over 350x300mm contact, medium mesh",
    "state": {"mode": "free", "params": {}, "messages": [], "user_provided_keys": [], "current_asking": null, "memory": null, "analysis_generated": false, "analysis_vtk_file": null, "analysis_plot_files": []},
    "llm_config": {"ollama_url": "http://localhost:11434", "model": "llama3.2", "api_key": null}
  }' | python -m json.tool
```

Expected: `state.params` should contain `a=5000, b=3000, t=250, Emod≈27386, q≈0.762, x1=0, x2=350, y1=0, y2=300` and `x`/`y` arrays with 16 nodes each.

- [ ] **Step 4: Commit the final state**

```bash
git add -A
git commit -m "feat: complete LLM-only extraction and VTK routing — remove all deterministic parsing"
```

---

## Self-Review Checklist

**Spec coverage:**
- §2.1 delete list → Tasks 7 (agent_logic) and 9 (vtk) ✓
- §2.2 new LLM call points → Tasks 3+4+5 (extraction) and 8 (VTK) ✓
- §3 expert system prompt → Task 3 ✓
- §3.3 computation formulas → embedded in system prompt (Task 3) ✓
- §3.4 multi-wheel load_cases → rules in system prompt, tested in Task 1 ✓
- §3.5 clarification rules → in system prompt, tested in Task 1 ✓
- §4.1 extraction schema → `_normalize_result` in Task 5 ✓
- §4.2 VTK router schema → `_parse_agentic_intent` in Task 8 ✓
- §5.1 clarification flow → main.py already handles `needs_clarification` ✓
- §5.2 JSON parse failure → retry logic in Task 5 ✓
- §6 main.py changes → Task 10 ✓

**No placeholders:** All steps include actual code. ✓

**Type consistency:** `extracted_multiple` key used in `main.py` is populated by `_normalize_result` in every task. `LookupIntent` fields match existing dataclass definition. ✓
