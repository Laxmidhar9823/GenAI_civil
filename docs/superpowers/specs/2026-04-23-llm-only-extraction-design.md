# LLM-Only Parameter Extraction & VTK Routing — Design Spec

**Date:** 2026-04-23  
**Branch:** autonomous_bot  
**Goal:** Replace all deterministic regex/heuristic parsing in `agent_logic.py` and `vtk_lookup_agent.py` with a pure LLM-driven approach. Priority is correctness; cost is irrelevant. Full stack remains on Ollama.

---

## 1. Problem Statement

The current system uses deterministic parsing first and consults the LLM only as a last resort. This causes misclassification — e.g. `5000×2000` typed as a contact area is matched by the slab dimension regex before the contact area pattern runs, producing wrong values silently. Naive users cannot be expected to phrase inputs in ways that satisfy regex order.

---

## 2. Architecture

### 2.1 What Is Deleted

**`backend/agent_logic.py`** (~1,400 lines removed):
- `_normalize_text`, `_contains_default_keyword`, `_looks_like_greeting/thanks/goodbye/help/explanation`
- `handle_conversational_intent`
- `infer_semantic_choices`
- `parse_multi_param_candidates`, `_parse_node_list_candidates`
- `_extract_engineering_candidates`, `_extract_assumed_values`
- `_extract_first_number_with_unit`, `_extract_all_numbers_with_units`
- `_tokenize_for_numbers`, `_parse_number_token_with_unit`
- `_find_phrase_indices`, `_extract_param_key_from_text`
- `_compute_patch_from_description`, `_build_group_wheel_load_cases`
- `infer_nodes_from_mesh`, `infer_load_patch_from_location`, `infer_load_patch_from_contact_area`
- `apply_implicit_inferences`
- `_extract_spacing_mm`, `_extract_wheel_spacing_mm`, `_extract_axle_spacing_mm`
- `_detect_number_of_wheels`, `_extract_edge_distances_mm`
- `_offset_load_patch`, `_clamp_center_to_slab`, `_center_to_bounded_patch`
- `_reference_point_from_token`, `_first_wheel_center_from_token`, `_dual_spacing_direction`
- `_resolve_reference_location_token`
- `normalize_mesh_type`, `normalize_load_location` (no longer needed externally)
- `PARAM_ALIASES`, `SINGLE_DEFAULT_PATTERNS`, `_GREETINGS`, `_THANKS`, `_GOODBYES`, `_HELP_PATTERNS`, `_EXPLAIN_PATTERNS`
- `MESH_LEVEL_ELEMENTS`
- All `re` import usage (module import removed)
- `infer_semantic_choices`, `_detect_extraction_ambiguity`

**`backend/vtk_lookup_agent.py`** (~300 lines removed):
- `_FIELD_ALIASES`, `_AGG_KEYWORDS`, `_DETAIL_HINTS`, `_FLEXURAL_HINTS`, `_IMPLICIT_RESULT_HINTS`, `_PLOT_REQUEST_HINTS`, `_GENERIC_STRESS_TERMS`, `_QUALITATIVE_ANALYSIS_TERMS`, `_FIELD_LISTING_TERMS`
- `_detect_aggregation`, `_detect_component`, `_detect_field`
- `_is_lookup_intent`, `_is_plot_request`
- `_parse_intent` (deterministic path — fully removed)
- `_default_aggregate_field_for_query`, `_coerce_aggregation`
- All `re` import usage

### 2.2 What Replaces It

Two new LLM call points:

**A. `process_user_input_with_llm()`** — redesigned to make a single LLM call with:
- `create_expert_system_prompt()` — rich domain-knowledge system prompt (see §3)
- `create_extraction_prompt()` — user input + conversation context + current params + current_asking
- Output: structured JSON with `understood` and `computed` sections (see §4)

**B. VTK intent router** — `_parse_agentic_intent()` becomes the only path in `handle_vtk_lookup_command`. The deterministic `_parse_intent()` fallback is removed entirely.

### 2.3 What Is Kept Unchanged

- `validate_single_param`, `check_physical_feasibility`, `find_first_inconsistent_param` — validation, not parsing
- `build_final_configuration`, `generate_completion_message`, `generate_autonomous_assistant_message`, `generate_vtk_agent_response` — narration layer
- All VTK script runners (`_run_script` + the three scripts) — unchanged
- `main.py` state machine, API endpoints, merge logic — minimal changes only

---

## 3. Expert System Prompt (`create_expert_system_prompt`)

The system prompt is a single authoritative brief passed on every extraction call. It contains:

### 3.1 Identity
> You are a rigid pavement FEM configuration assistant with expert knowledge of IS 456, IRC:58, and pavement engineering. You extract user-provided parameters and compute ALL derived values yourself using the formulas below. You never guess or assume ambiguous inputs — if anything is unclear, you ask exactly one clarification question and wait.

### 3.2 Parameter Catalogue
Full catalogue of all parameters (`a`, `b`, `t`, `Emod`, `nu`, `Kx`, `Ky`, `Kz`, `x1`, `x2`, `y1`, `y2`, `q`, `mesh_type`, `load_location`, `x[]`, `y[]`, `load_cases[]`), each entry including:
- Technical name and unit
- Typical range for rigid pavements
- Disambiguation note: common phrasings that could be confused with another parameter

**Key disambiguation rules embedded in the catalogue:**
- "If user gives a dimension pair (NNN×NNN) without the word 'contact', 'tyre', or 'tire', do NOT assume it is contact area. Ask: 'Is NNN×NNN mm the slab size or the tyre contact dimensions?'"
- "If user says 'slab 5m × 3m', that is unambiguously a=5000mm, b=3000mm."
- "Contact area requires explicit context: 'contact area', 'tyre contact', 'over NNN×NNN', 'contact patch'."

### 3.3 Mandatory Computation Rules

| Quantity | Formula | Example |
|---|---|---|
| Elastic modulus from grade | IS 456: E = 5000 × √fck (MPa), fck in MPa | M30 → E = 5000×√30 = 27,386 MPa |
| Contact pressure q | q (MPa) = P(kN)×1000 / (Lx_mm × Ly_mm) | 80 kN, 350×300 mm → q = 80000/105000 = 0.7619 MPa |
| Poisson ratio default | ν = 0.15 per IRC:58 unless user specifies | — |
| Subgrade K isotropy | Single K value → Kx=Ky=Kz unless differentiated | K=60 → Kx=Ky=Kz=60 |
| Node arrays — coarse | 11 nodes: x[i] = i×(a/10), i=0..10; same for y | a=5000 → [0,500,1000,...,5000] |
| Node arrays — medium | 16 nodes: x[i] = i×(a/15), i=0..15 | — |
| Node arrays — fine | 31 nodes: x[i] = i×(a/30), i=0..30 | — |
| Load patch — corner | x1=0, y1=0, x2=Lx, y2=Ly | — |
| Load patch — edge x=0 | x1=0, x2=Lx, y1=(b−Ly)/2, y2=(b+Ly)/2 | — |
| Load patch — edge x=a | x1=a−Lx, x2=a, y1=(b−Ly)/2, y2=(b+Ly)/2 | — |
| Load patch — edge y=0 | x1=(a−Lx)/2, x2=(a+Lx)/2, y1=0, y2=Ly | — |
| Load patch — edge y=b | x1=(a−Lx)/2, x2=(a+Lx)/2, y1=b−Ly, y2=b | — |
| Load patch — interior | x1=(a−Lx)/2, x2=(a+Lx)/2, y1=(b−Ly)/2, y2=(b+Ly)/2 | — |
| Load patch — "edge" (unspecified) | Ask: "Which edge — x=0, x=a, y=0, or y=b?" | Triggers clarification |
| Per-wheel q (axle load) | q = (P_total/N_wheels)×1000 / (Lx×Ly) | — |
| Dual tyre patch centers | y_left = b/2 − s/2, y_right = b/2 + s/2 | s = wheel spacing |
| Tandem axle patch centers | x_front = a/2 − d/2, x_rear = a/2 + d/2 | d = axle spacing |

### 3.4 Multi-Wheel `load_cases` Array
Each entry in `load_cases` is `{x1, x2, y1, y2, q}` for one tyre contact patch. Rules:
- Single wheel: `load_cases` is empty `[]`; patch coordinates go directly in `computed` as `x1/x2/y1/y2`
- Dual tyres: 2 entries, side by side across slab width
- Tandem axle (single tyre/axle): 2 entries along slab length
- Tandem axle + dual tyres: 4 entries (2 axles × 2 tyres)
- First entry is always also copied to top-level `x1/x2/y1/y2/q` for backward compatibility

### 3.5 Clarification Rules
The LLM must ask for clarification (set `needs_clarification=true`) when:
- A dimension pair is given without keyword context (slab vs. contact ambiguity)
- A load value is given but no contact dimensions are provided (cannot compute q)
- Load location is not specified and cannot be inferred from context
- User mentions "edge" without specifying which edge (if the distinction matters for the specific slab geometry)
- Contradictory values are given (e.g., contact area larger than slab)

The LLM must NOT ask for clarification when:
- A value can be computed from what is already provided (e.g., E from M-grade, q from load+contact)
- A standard default exists per IRC/IS and user has shown no preference
- The value is implied by context (e.g., "interior load" → no edge ambiguity)

---

## 4. Output JSON Schema

### 4.1 Extraction Response (from `process_user_input_with_llm`)

```json
{
  "needs_clarification": false,
  "clarification_question": null,
  "friendly_response": "Got it — M30 slab 5×3 m, 250 mm thick, K=60 MPa/m, 80 kN wheel at edge over 350×300 mm contact. I computed E=27,386 MPa, q=0.762 MPa, and placed the load patch at the x=0 edge centred across the width.",
  "use_default": false,
  "use_all_defaults": false,
  "understood": {
    "slab_a_mm": 5000,
    "slab_b_mm": 3000,
    "thickness_mm": 250,
    "fck_mpa": 30,
    "subgrade_k": 60,
    "load_kn": 80,
    "contact_lx_mm": 350,
    "contact_ly_mm": 300,
    "load_location": "edge",
    "mesh_type": "medium"
  },
  "computed": {
    "a": 5000,
    "b": 3000,
    "t": 250,
    "Emod": 27386.13,
    "nu": 0.15,
    "Kx": 60,
    "Ky": 60,
    "Kz": 60,
    "q": 0.7619,
    "x1": 0,
    "x2": 350,
    "y1": 1325,
    "y2": 1625,
    "mesh_type": "medium",
    "load_location": "edge",
    "x": [0, 333.33, 666.67, 1000, 1333.33, 1666.67, 2000, 2333.33, 2666.67, 3000, 3333.33, 3666.67, 4000, 4333.33, 4666.67, 5000],
    "y": [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
    "load_cases": []
  }
}
```

When `needs_clarification=true`:
```json
{
  "needs_clarification": true,
  "clarification_question": "Is 5000×2000 mm the slab size or the tyre contact dimensions?",
  "friendly_response": "Just to confirm — is 5000×2000 mm your slab size, or is it the tyre contact area?",
  "use_default": false,
  "use_all_defaults": false,
  "understood": {},
  "computed": {}
}
```

### 4.2 VTK Router Response (from `_parse_agentic_intent`)

```json
{
  "is_vtk_query": true,
  "action": "aggregate",
  "field": "sxx_top",
  "aggregation": "max",
  "component": null,
  "use_abs": false,
  "with_plot": false
}
```

---

## 5. Clarification & Error Flow

### 5.1 Clarification
- `main.py` checks `result["needs_clarification"]`; if true, the assistant response is `result["friendly_response"]` and `state.params` is NOT updated
- The conversation history carries the clarification exchange; next user turn resolves it with full context
- `current_asking` is unchanged between turns during clarification

### 5.2 JSON Parse Failure
1. Strip markdown fences, find `{...}` substring — retry parse
2. If still fails: retry the LLM call once with a suffix appended to the prompt: `"Return ONLY the JSON object. No explanation, no markdown, no prose."`
3. If second attempt fails: return `{"needs_clarification": true, "friendly_response": "I didn't quite catch that — could you rephrase?", ...}`

### 5.3 Timeout
Unchanged from current implementation — friendly message, no state update.

### 5.4 VTK Router Failure
If `_parse_agentic_intent` returns `None` (timeout, parse failure, or `is_vtk_query=false`), `handle_vtk_lookup_command` returns `None` and `main.py` falls through to `generate_vtk_agent_response` — unchanged from current behaviour.

---

## 6. Changes to `main.py`

| Location | Change |
|---|---|
| Import block | Remove `apply_implicit_inferences`, `normalize_load_location`, `normalize_mesh_type`, `infer_semantic_choices` imports |
| `~line 816` | Remove `apply_implicit_inferences(state.params, state.user_provided_keys)` call and inference_notes injection |
| `~line 584` result merge | `extracted_multiple` is now `result.get("computed", {})` — update key reference |
| VTK lookup call | Remove `prefer_agentic=False` default; always use agentic path |

No changes to state machine logic, API endpoints, or response formatting.

---

## 7. Files Modified

| File | Change type |
|---|---|
| `backend/agent_logic.py` | Delete ~1,400 lines of deterministic code; rewrite `process_user_input_with_llm`, `create_expert_system_prompt`, `create_extraction_prompt` |
| `backend/vtk_lookup_agent.py` | Delete ~300 lines of deterministic helpers; clean up `_parse_agentic_intent`; remove `_parse_intent` fallback from `handle_vtk_lookup_command` |
| `backend/main.py` | Remove `apply_implicit_inferences` call; update result key reference; always use agentic VTK path |

---

## 8. Out of Scope

- Switching LLM backend (stays Ollama)
- Changing the FEM solver or VTK scripts
- UI changes
- `generate_autonomous_assistant_message` or narration prompts
