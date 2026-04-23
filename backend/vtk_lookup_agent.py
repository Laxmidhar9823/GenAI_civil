"""VTK analysis agent — kimi multimodal LLM is the sole router and responder.

All deterministic keyword/regex routing has been removed.  The flow is:

1. Route: ask kimi (text-only, fast) which script action to take.
2. Execute: run the chosen script to get structured numeric data.
3. Respond: ask kimi (multimodal, with images) to answer the user using
   both the structured data and the generated contour plots.

handle_vtk_lookup_command returns None when:
  - No analysis has been run yet.
  - kimi is unavailable (no URL/model configured).
  - kimi classifies the question as NOT about analysis results.
Callers should fall through to the normal chat handler in those cases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.vtk_stats import describe_dataset
from backend.ollama import invoke_ollama_multimodal_chat


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VTK_FILE = REPO_ROOT / "output_python_square" / "results.vtk"
DETAIL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_detailed_report.py"
AGG_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_aggregate_query.py"
FLEXURAL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_flexural_report.py"

_ROUTING_TIMEOUT = 12.0
_RESPONSE_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke_llm_with_timeout(llm: ChatOllama, messages: List, timeout_seconds: float = 12.0):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(llm.invoke, messages)
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise TimeoutError(f"LLM request timed out after {timeout_seconds:.1f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _safe_parse_json_object(text: str) -> Optional[Dict]:
    body = (text or "").strip()
    if not body:
        return None
    if body.startswith("```"):
        first_newline = body.find("\n")
        if first_newline != -1:
            body = body[first_newline + 1:]
        if body.endswith("```"):
            body = body[:-3]
        body = body.strip()
    start = body.find("{")
    end = body.rfind("}")
    if start != -1 and end != -1 and end > start:
        body = body[start: end + 1]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _run_script(script_path: Path, args: List[str]) -> Dict:
    cmd = [sys.executable, str(script_path), *args]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "Unknown script error."
        raise RuntimeError(detail)
    payload_text = completed.stdout.strip()
    if not payload_text:
        raise RuntimeError("The VTK query script did not return any output.")
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The VTK query script returned invalid JSON.") from exc


def _resolve_plot_paths(plot_urls: List[str]) -> List[Path]:
    artifacts_dir = REPO_ROOT / "output_python_square"
    resolved: List[Path] = []
    for raw in plot_urls or []:
        text = (raw or "").strip()
        if not text:
            continue
        candidate: Optional[Path] = None
        if text.startswith("http://") or text.startswith("https://"):
            parsed = urlparse(text)
            basename = Path(parsed.path).name
            if basename:
                candidate = artifacts_dir / basename
        elif text.startswith("/artifacts/"):
            candidate = artifacts_dir / Path(text).name
        else:
            p = Path(text)
            candidate = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        if candidate and candidate.exists() and candidate.is_file():
            resolved.append(candidate)
    seen: set = set()
    unique: List[Path] = []
    for p in resolved:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# VTK stats summary (used by main.py and the routing prompt)
# ---------------------------------------------------------------------------

def build_vtk_stats_summary(vtk_file_path: str) -> str:
    """Return a compact text summary of all result fields for LLM context."""
    try:
        info = describe_dataset(vtk_file_path)
    except Exception as exc:
        return f"(Could not load VTK stats: {exc})"

    lines = [
        f"**Mesh:** {info.get('num_points', '?')} nodes, {info.get('num_cells', '?')} elements",
        "",
        "**Result fields (point data):**",
        "| Field | Min | Max | Mean |",
        "|-------|-----|-----|------|",
    ]
    for field in info.get("point_data_fields", []):
        name = field.get("name", "?")
        stats = field.get("component_stats", [{}])[0]

        def _fmt(v):
            return f"{v:.4g}" if v is not None else "N/A"

        mn = stats.get("min")
        mx = stats.get("max")
        me = stats.get("mean")
        lines.append(f"| {name} | {_fmt(mn)} | {_fmt(mx)} | {_fmt(me)} |")

    cell_fields = info.get("available_fields", {}).get("cell_data", [])
    if cell_fields:
        lines.append(f"\n**Cell fields:** {', '.join(cell_fields)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured result formatters (used as LLM input context)
# ---------------------------------------------------------------------------

def _format_flexural_response(payload: Dict, api_base_url: str = "") -> str:
    results = payload.get("results", {})
    lines = [
        "### Flexural Stress Report",
        "",
        "| Field | Surface | Max (MPa) | Min (MPa) | Mean (MPa) |",
        "|-------|---------|-----------|-----------|------------|",
    ]
    for field_name, label, surface in [
        ("sxx_top", "σ_x", "Top"),
        ("sxx_bottom", "σ_x", "Bottom"),
        ("syy_top", "σ_y", "Top"),
        ("syy_bottom", "σ_y", "Bottom"),
    ]:
        r = results.get(field_name)
        if r is None or not r.get("available", False):
            lines.append(f"| {label} | {surface} | N/A | N/A | N/A |")
            continue

        def _v(k):
            v = r.get(k)
            return f"{v:.6g}" if v is not None else "N/A"

        lines.append(f"| {label} | {surface} | {_v('max')} | {_v('min')} | {_v('mean')} |")

    missing = payload.get("missing_fields", [])
    if missing:
        lines.append(f"\n> Missing fields: {', '.join(missing)}")

    contour_plots = payload.get("contour_plots") or []
    if contour_plots:
        lines.append("\n**Stress Contour Plots:**")
        for p in contour_plots:
            basename = Path(p).name
            url = f"{api_base_url}/artifacts/{basename}" if api_base_url else f"/artifacts/{basename}"
            lines.append(f"![{basename.replace('_', ' ').replace('.png', '')}]({url})")

    return "\n".join(lines)


def _format_detail_response(payload: Dict) -> str:
    point_fields = payload.get("available_fields", {}).get("point_data", [])
    cell_fields = payload.get("available_fields", {}).get("cell_data", [])
    lines = [
        "### Analysis Results Overview",
        f"- Mesh: {payload.get('num_points', 0)} nodes, {payload.get('num_cells', 0)} elements",
        f"- Point fields: {', '.join(point_fields) if point_fields else '(none)'}",
        f"- Cell fields: {', '.join(cell_fields) if cell_fields else '(none)'}",
    ]
    return "\n".join(lines)


def _format_aggregate_response(payload: Dict) -> str:
    value = payload.get("value")
    value_text = f"{value:.6g}" if isinstance(value, float) else str(value)
    field = payload.get("resolved_field", payload.get("requested_field", "unknown"))
    agg = payload.get("aggregation", "unknown")
    component = payload.get("component", "scalar")
    n = payload.get("sample_size", "unknown")
    return (
        f"### Query Result\n"
        f"- Field: **{field}**\n"
        f"- Aggregation: {agg}\n"
        f"- Component: {component}\n"
        f"- Sample size: {n}\n"
        f"- **Value: {value_text}**"
    )


# ---------------------------------------------------------------------------
# kimi routing (decides which script to run)
# ---------------------------------------------------------------------------

_ROUTING_SYSTEM = (
    "You are a pavement VTK analysis router. "
    "Classify the user's question and choose the best data-retrieval action. "
    "Return strict JSON only. No explanation."
)

_ROUTING_ACTIONS = """Actions:
- "flexural"  — bending / flexural stress report (sxx_top, sxx_bottom, syy_top, syy_bottom)
- "detail"    — list available result fields and mesh metadata
- "aggregate" — compute one statistic for one field (include "field" and "agg" keys)
    Fields: w, u, v, theta_x, theta_y, sxx_top, syy_top, sxy_top,
            sxx_bottom, syy_bottom, sxy_bottom, sxx_membrane, syy_membrane, sxy_membrane
    Aggregations: max, min, mean, sum, std, median, count, p95, p05
- "plots"     — user wants to see / inspect the generated contour images
- "answer"    — answer directly from the VTK summary (no script needed)
- "not_vtk"   — NOT about analysis results (param changes, general questions)

Return:
{
  "action": "<one of the above>",
  "field": "<field key or null>",
  "agg": "<aggregation or null>",
  "reason": "<one short sentence>"
}"""


def _route_with_kimi(
    *,
    llm: ChatOllama,
    user_input: str,
    vtk_stats: str,
    available_plot_names: List[str],
) -> Optional[Dict]:
    plots_line = (
        "Generated plots: " + ", ".join(available_plot_names)
        if available_plot_names
        else "No plots generated yet."
    )
    prompt = (
        f"VTK ANALYSIS SUMMARY:\n{vtk_stats}\n\n"
        f"{plots_line}\n\n"
        f"{_ROUTING_ACTIONS}\n\n"
        f"USER QUESTION: {user_input}"
    )
    try:
        response = _invoke_llm_with_timeout(
            llm,
            [SystemMessage(content=_ROUTING_SYSTEM), HumanMessage(content=prompt)],
            timeout_seconds=_ROUTING_TIMEOUT,
        )
        content = str(getattr(response, "content", "") or "")
        return _safe_parse_json_object(content)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Script execution helpers
# ---------------------------------------------------------------------------

def _execute_action(
    action: str,
    field: Optional[str],
    agg: Optional[str],
    vtk_path: Path,
    api_base_url: str,
) -> tuple[str, List[Path]]:
    """Run the appropriate script and return (formatted_result, extra_plot_paths)."""
    extra_plots: List[Path] = []

    if action == "flexural":
        try:
            payload = _run_script(FLEXURAL_SCRIPT, ["--file", str(vtk_path)])
            formatted = _format_flexural_response(payload, api_base_url)
            contour_urls = [
                f"{api_base_url}/artifacts/{Path(p).name}"
                for p in (payload.get("contour_plots") or [])
            ]
            extra_plots = _resolve_plot_paths(contour_urls)
            return formatted, extra_plots
        except Exception as e:
            return f"Flexural script error: {e}", []

    if action == "detail":
        try:
            payload = _run_script(DETAIL_SCRIPT, ["--file", str(vtk_path)])
            return _format_detail_response(payload), []
        except Exception as e:
            return f"Detail script error: {e}", []

    if action == "aggregate":
        f = field or "w"
        a = agg or "max"
        try:
            payload = _run_script(AGG_SCRIPT, ["--file", str(vtk_path), "--field", f, "--agg", a])
            return _format_aggregate_response(payload), []
        except Exception as e:
            return f"Aggregate script error: {e}", []

    # "answer", "plots", or unknown → no script needed
    return "", []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_RESPONSE_SYSTEM = (
    "You are an expert rigid pavement analysis assistant. "
    "Answer the user's question using the analysis data provided. "
    "Be concise, engineering-focused, and cite exact numeric values. "
    "When plot images are attached, describe what you observe — stress patterns, "
    "deflection distribution, concentration zones, etc. "
    "Use Markdown tables or bullets where helpful. "
    "Do not output JSON. Do not mention file paths."
)


def handle_vtk_lookup_command(
    user_input: str,
    *,
    default_vtk_file: Optional[str] = None,
    allow_implicit: bool = False,
    api_base_url: str = "",
    llm: Optional[ChatOllama] = None,
    prefer_agentic: bool = False,  # kept for signature compatibility; kimi is always used
    available_plot_urls: Optional[List[str]] = None,
    narration_url: Optional[str] = None,
    narration_model: Optional[str] = None,
    narration_api_key: Optional[str] = None,
) -> Optional[str]:
    """Kimi-driven VTK query handler.

    Returns None when:
    - No analysis exists yet (vtk file missing).
    - kimi is not configured.
    - kimi classifies the question as not about analysis results.

    Returns a response string for all VTK-related questions.
    """
    # --- Guard: need analysis and kimi ---
    vtk_path = (
        Path(default_vtk_file).resolve()
        if default_vtk_file
        else DEFAULT_VTK_FILE.resolve()
    )
    if not vtk_path.exists():
        return None

    if not allow_implicit:
        return None

    if not (narration_url and narration_model):
        return None

    if llm is None:
        return None

    # --- Step 1: gather VTK context ---
    vtk_stats = build_vtk_stats_summary(str(vtk_path))
    plot_urls = [u for u in (available_plot_urls or []) if isinstance(u, str) and u.strip()]
    local_plots = _resolve_plot_paths(plot_urls)
    available_plot_names = [Path(u).name for u in plot_urls]

    # --- Step 2: route with kimi (text-only, fast) ---
    route = _route_with_kimi(
        llm=llm,
        user_input=user_input,
        vtk_stats=vtk_stats,
        available_plot_names=available_plot_names,
    )

    if route is None:
        # kimi routing failed — don't return a stale result, let main handler take over
        return None

    action = str(route.get("action", "not_vtk")).strip().lower()

    if action == "not_vtk":
        return None  # parameter change or general question → main chat handler

    # --- Step 3: execute script if needed ---
    structured_data = vtk_stats  # default context: always include stats
    extra_plot_paths: List[Path] = []

    if action in ("flexural", "detail", "aggregate"):
        field = route.get("field") or None
        agg = route.get("agg") or None
        result_text, extra_plot_paths = _execute_action(
            action, field, agg, vtk_path, api_base_url
        )
        if result_text:
            structured_data = result_text + "\n\n---\n\n**Full VTK stats:**\n" + vtk_stats

    # Combine all available plots (script-generated + existing)
    all_plots = extra_plot_paths + local_plots
    # Deduplicate
    seen: set = set()
    deduped_plots: List[Path] = []
    for p in all_plots:
        k = str(p)
        if k not in seen:
            seen.add(k)
            deduped_plots.append(p)

    # --- Step 4: kimi multimodal response ---
    response_prompt = (
        f"### USER QUESTION\n{user_input}\n\n"
        f"### ANALYSIS DATA\n{structured_data}\n\n"
        "Provide a thorough engineering answer. "
        "Reference exact values. "
        "If contour plot images are attached, analyse what you observe visually — "
        "identify stress concentration zones, deflection patterns, critical regions."
    )

    try:
        response_text = invoke_ollama_multimodal_chat(
            base_url=narration_url,
            model=narration_model,
            system_prompt=_RESPONSE_SYSTEM,
            user_prompt=response_prompt,
            image_paths=deduped_plots,
            api_key=narration_api_key,
            timeout_seconds=_RESPONSE_TIMEOUT,
        )
        cleaned = (response_text or "").strip()
        if cleaned:
            return cleaned
    except Exception:
        pass

    # Fallback: return structured data if kimi response fails
    if structured_data and structured_data != vtk_stats:
        return structured_data

    return None
