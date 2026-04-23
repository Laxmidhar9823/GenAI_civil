from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.vtk_stats import describe_dataset
from backend.ollama import invoke_ollama_multimodal_chat


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VTK_FILE = REPO_ROOT / "output_python_square" / "results.vtk"
DETAIL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_detailed_report.py"
AGG_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_aggregate_query.py"
FLEXURAL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_flexural_report.py"


@dataclass
class LookupIntent:
    action: str
    vtk_file: Path
    aggregation: Optional[str] = None
    field: Optional[str] = None
    component: Optional[str] = None
    use_abs: bool = False
    with_plot: bool = False
    field_explicit: bool = True  # False when field was a fallback default, not user-specified


def _safe_parse_json_object(text: str) -> Optional[Dict]:
    body = (text or "").strip()
    if not body:
        return None

    if body.startswith("```"):
        first_newline = body.find("\n")
        if first_newline != -1:
            body = body[first_newline + 1 :]
        if body.endswith("```"):
            body = body[:-3]
        body = body.strip()

    start = body.find("{")
    end = body.rfind("}")
    if start != -1 and end != -1 and end > start:
        body = body[start : end + 1]

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _invoke_llm_with_timeout(llm: ChatOllama, messages: List, timeout_seconds: float = 10.0):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(llm.invoke, messages)
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise TimeoutError(f"LLM request timed out after {timeout_seconds:.1f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _extract_vtk_path(text: str, default_vtk_file: Optional[Path] = None) -> Path:
    # Accept quoted and unquoted paths ending with a known VTK-like extension.
    match = re.search(r"([\w./\\-]+\.(?:vtk|vtu|vtp|vtr|vts|vti))", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().strip("'\"")
        path = Path(candidate)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path
    if default_vtk_file is not None:
        return default_vtk_file.resolve()
    return DEFAULT_VTK_FILE.resolve()


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
            "w",
            "u",
            "v",
            "theta_x",
            "theta_y",
            "sxx_top",
            "sxx_bottom",
            "syy_top",
            "syy_bottom",
            "sxy_top",
            "sxy_bottom",
            "sxx_membrane",
            "syy_membrane",
            "sxy_membrane",
            "displacement",
        ])

    return (
        f"allow_implicit={str(bool(allow_implicit)).lower()}\n"
        f"vtk_file={vtk_file}\n"
        f"AVAILABLE_FIELDS={', '.join(available_fields)}\n\n"
        f"USER_QUERY: {user_input}\n\n"
        "Return the JSON object."
    )


def _build_vtk_router_system_prompt() -> str:
    return (
        "You are a VTK result query router for a rigid pavement FEM analysis tool.\n"
        "Classify the user's query and return STRICT JSON only - no prose, no explanation.\n\n"
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
        "- flexural: user asks about bending/flexural stress, top/bottom surface stress, sigma_x, sigma_y.\n"
        "- plots: user asks to see/show/view generated contour plots or images.\n"
        "- none: query is unrelated to FEM/VTK results.\n\n"
        "FIELD MAPPING (use exact canonical names):\n"
        "- deflection / vertical displacement / out-of-plane displacement / w -> field='w'\n"
        "- x displacement / longitudinal displacement / u -> field='u'\n"
        "- y displacement / transverse displacement / v -> field='v'\n"
        "- rotation about y / slope x / theta_x -> field='theta_x'\n"
        "- rotation about x / slope y / theta_y -> field='theta_y'\n"
        "- sigma_x top / sxx top / top normal stress x -> field='sxx_top'\n"
        "- sigma_x bottom / sxx bottom -> field='sxx_bottom'\n"
        "- sigma_y top / syy top -> field='syy_top'\n"
        "- sigma_y bottom / syy bottom -> field='syy_bottom'\n"
        "- shear top / tau_xy top / sxy top -> field='sxy_top'\n"
        "- shear bottom / sxy bottom -> field='sxy_bottom'\n"
        "- If user asks about 'stress' generally -> route to flexural (action='flexural', field=null)\n"
        "- If field is unclear for aggregate -> set field=null (will route to flexural)\n\n"
        "AGGREGATION MAPPING:\n"
        "- maximum / highest / peak / max -> 'max'\n"
        "- minimum / lowest / min -> 'min'\n"
        "- average / mean / avg -> 'mean'\n"
        "- standard deviation / std / stdev -> 'std'\n"
        "- total / sum -> 'sum'\n"
        "- median -> 'median'\n"
        "- 95th percentile / p95 -> 'p95'\n"
        "- 5th percentile / p05 -> 'p05'\n\n"
        "allow_implicit=true means treat implicit result references (deflection, stress) "
        "as VTK queries even without the word 'vtk'.\n"
    )


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
            "w",
            "u",
            "v",
            "theta_x",
            "theta_y",
            "sxx_top",
            "sxx_bottom",
            "syy_top",
            "syy_bottom",
            "sxy_top",
            "sxy_bottom",
            "sxx_membrane",
            "syy_membrane",
            "sxy_membrane",
            "displacement",
        ]

    system = SystemMessage(content=_build_vtk_router_system_prompt())
    human = HumanMessage(
        content=_build_planner_prompt(
            user_input=user_input,
            vtk_file=vtk_file,
            allow_implicit=allow_implicit,
        )
    )

    try:
        response = _invoke_llm_with_timeout(llm, [system, human], timeout_seconds=10.0)
        content = str(getattr(response, "content", "") or "")
        payload = _safe_parse_json_object(content)
    except Exception:
        return None

    if not payload:
        return None

    if not bool(payload.get("is_vtk_query")):
        return None

    action = " ".join(str(payload.get("action") or "none").strip().lower().split())
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
        raise RuntimeError("The VTK query script returned invalid JSON output.") from exc


def build_vtk_stats_summary(vtk_file_path: str) -> str:
    """Return a compact text summary of all fields in a VTK file for LLM context."""
    try:
        info = describe_dataset(vtk_file_path)
    except Exception as exc:
        return f"(Could not load VTK stats: {exc})"

    lines = [
        f"**Mesh:** {info.get('num_points', '?')} nodes, {info.get('num_cells', '?')} elements",
        "",
        "**Result fields:**",
        "| Field | Min | Max | Mean |",
        "|-------|-----|-----|------|",
    ]

    for field in info.get("point_data_fields", []):
        name = field.get("name", "?")
        stats = field.get("component_stats", [{}])[0]
        mn = stats.get("min")
        mx = stats.get("max")
        me = stats.get("mean")

        def _fmt(v):
            return f"{v:.4g}" if v is not None else "N/A"

        lines.append(f"| {name} | {_fmt(mn)} | {_fmt(mx)} | {_fmt(me)} |")

    cell_fields = info.get("available_fields", {}).get("cell_data", [])
    if cell_fields:
        lines.append(f"\n**Cell fields:** {', '.join(cell_fields)}")

    return "\n".join(lines)


def _format_flexural_response(payload: Dict, api_base_url: str = "") -> str:
    results = payload.get("results", {})
    missing = payload.get("missing_fields", [])

    lines = [
        "### Flexural Stress Analysis",
        "",
        "| Field | Surface | Statistic | Value (MPa) |",
        "|-------|---------|-----------|-------------|",
    ]

    field_order = ["sxx_top", "sxx_bottom", "syy_top", "syy_bottom"]
    labels = {
        "sxx_top": ("σ_x", "Top"),
        "sxx_bottom": ("σ_x", "Bottom"),
        "syy_top": ("σ_y", "Top"),
        "syy_bottom": ("σ_y", "Bottom"),
    }
    stat_specs = [("max", "Max"), ("min", "Min"), ("mean", "Mean")]

    for field_name in field_order:
        r = results.get(field_name)
        if r is None:
            continue
        label, surface = labels[field_name]
        if not r.get("available", False):
            for _, stat_label in stat_specs:
                lines.append(f"| {label} | {surface} | {stat_label} | N/A |")
            continue
        for stat_key, stat_label in stat_specs:
            val = r.get(stat_key)
            val_str = f"{val:.6g}" if val is not None else "N/A"
            lines.append(f"| {label} | {surface} | {stat_label} | {val_str} |")

    if missing:
        lines.append(f"\n> Missing fields: {', '.join(missing)}")

    contour_plots = payload.get("contour_plots")
    if contour_plots:
        lines.append("\n**Stress Contour Plots:**")
        for p in contour_plots:
            basename = Path(p).name
            url = f"{api_base_url}/artifacts/{basename}" if api_base_url else f"/artifacts/{basename}"
            label = basename.replace("_", " ").replace(".png", "")
            lines.append(f"![{label}]({url})")

    return "\n".join(lines)


def _format_plot_gallery(plot_urls: List[str]) -> str:
    if not plot_urls:
        return "No generated plots are available yet."

    lines = [
        "### Generated Analysis Plots",
        "",
    ]
    for idx, url in enumerate(plot_urls, start=1):
        lines.append(f"![Plot {idx}]({url})")
    return "\n".join(lines)


def _format_detail_response(payload: Dict) -> str:
    point_fields = payload.get("available_fields", {}).get("point_data", [])
    cell_fields = payload.get("available_fields", {}).get("cell_data", [])

    lines = [
        "### Analysis Results Overview",
        f"- Mesh: {payload.get('num_points', 0)} nodes, {payload.get('num_cells', 0)} elements",
    ]

    if point_fields:
        lines.append(f"- Point fields: {', '.join(point_fields)}")
    else:
        lines.append("- Point fields: (none)")

    if cell_fields:
        lines.append(f"- Cell fields: {', '.join(cell_fields)}")
    else:
        lines.append("- Cell fields: (none)")

    lines.append("\nAsk for aggregates like: mean/max/min of w, sxx_top, or any listed field.")
    return "\n".join(lines)


def _format_aggregate_response(payload: Dict) -> str:
    value = payload.get("value")
    if isinstance(value, float):
        value_text = f"{value:.6g}"
    else:
        value_text = str(value)

    field = payload.get("resolved_field", payload.get("requested_field", "unknown"))
    agg = payload.get("aggregation", "unknown")
    component = payload.get("component", "scalar")
    n = payload.get("sample_size", "unknown")

    lines = [
        "### Query Result",
        f"- Field: **{field}**",
        f"- Aggregation: {agg}",
        f"- Component: {component}",
        f"- Sample size: {n}",
        f"- **Value: {value_text}**",
    ]
    return "\n".join(lines)


_NARRATION_SYSTEM_PROMPT = (
    "You are an expert rigid pavement analysis assistant. "
    "Answer the user's question using the structured analysis result provided. "
    "Be concise and engineering-focused. Use Markdown tables or bullet points where helpful. "
    "Do not mention file paths or internal field names without context. "
    "Do not output JSON. When plot images are attached, incorporate observations from them."
)

_NARRATION_TIMEOUT = 45.0


def _resolve_plot_paths(plot_urls: List[str]) -> List[Path]:
    """Resolve plot URL strings to local filesystem paths for multimodal input."""
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


def _narrate_with_kimi(
    *,
    ollama_url: str,
    model: str,
    api_key: Optional[str],
    user_question: str,
    structured_data: str,
    local_plot_paths: Optional[List[Path]] = None,
    timeout: float = _NARRATION_TIMEOUT,
) -> Optional[str]:
    """Ask kimi to narrate a structured VTK result in natural engineering language."""
    user_prompt = (
        f"### USER QUESTION\n{user_question}\n\n"
        f"### ANALYSIS RESULT\n{structured_data}\n\n"
        "Provide a clear, professional engineering interpretation. "
        "Quote the exact numeric values from the result."
    )
    try:
        text = invoke_ollama_multimodal_chat(
            base_url=ollama_url,
            model=model,
            system_prompt=_NARRATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=local_plot_paths or [],
            api_key=api_key,
            timeout_seconds=timeout,
        )
        cleaned = (text or "").strip()
        return cleaned if cleaned else None
    except Exception:
        return None


def _narrate_plots_with_kimi(
    *,
    ollama_url: str,
    model: str,
    api_key: Optional[str],
    user_question: str,
    local_plot_paths: List[Path],
    plot_urls: List[str],
    timeout: float = _NARRATION_TIMEOUT,
) -> Optional[str]:
    """Ask kimi to visually analyse all generated plots."""
    if not local_plot_paths:
        return None
    plot_list = "\n".join(f"- {Path(u).name}" for u in plot_urls if u)
    user_prompt = (
        f"### USER QUESTION\n{user_question}\n\n"
        f"### AVAILABLE PLOTS\n{plot_list}\n\n"
        "The plots are attached as images. Provide a professional engineering analysis of "
        "the contour patterns, stress distributions, and deflection fields shown."
    )
    try:
        text = invoke_ollama_multimodal_chat(
            base_url=ollama_url,
            model=model,
            system_prompt=_NARRATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=local_plot_paths,
            api_key=api_key,
            timeout_seconds=timeout,
        )
        cleaned = (text or "").strip()
        return cleaned if cleaned else None
    except Exception:
        return None


_GLOBAL_STRESS_LABELS = {
    "sxx_top": ("σ_x", "Top"),
    "sxx_bottom": ("σ_x", "Bottom"),
    "syy_top": ("σ_y", "Top"),
    "syy_bottom": ("σ_y", "Bottom"),
}


def _run_global_stress_query(vtk_file: Path, aggregation: str) -> str:
    """Query all 4 flexural stress fields for the given aggregation.

    Runs AGG_SCRIPT for each of sxx_top, sxx_bottom, syy_top, syy_bottom,
    collects all values, and returns a formatted markdown table with all 4 results
    plus the overall winner (the field/surface producing the global max/min/mean).
    """
    agg_label = aggregation.upper()
    field_results: Dict[str, Optional[float]] = {}

    for field_name in ("sxx_top", "sxx_bottom", "syy_top", "syy_bottom"):
        try:
            payload = _run_script(AGG_SCRIPT, [
                "--file", str(vtk_file),
                "--field", field_name,
                "--agg", aggregation,
            ])
            val = payload.get("value")
            field_results[field_name] = float(val) if val is not None else None
        except Exception:
            field_results[field_name] = None

    lines = [
        f"### Global Stress Query — {agg_label}",
        "",
        "| Field | Surface | Value (MPa) |",
        "|-------|---------|-------------|",
    ]

    best_field: Optional[str] = None
    best_value: Optional[float] = None

    for field_name in ("sxx_top", "sxx_bottom", "syy_top", "syy_bottom"):
        label, surface = _GLOBAL_STRESS_LABELS[field_name]
        val = field_results.get(field_name)
        if val is not None:
            lines.append(f"| {label} | {surface} | {val:.6g} |")
            if best_value is None:
                best_value = val
                best_field = field_name
            elif aggregation == "max" and val > best_value:
                best_value = val
                best_field = field_name
            elif aggregation == "min" and val < best_value:
                best_value = val
                best_field = field_name
            elif aggregation == "mean" and abs(val) > abs(best_value):
                best_value = val
                best_field = field_name
        else:
            lines.append(f"| {label} | {surface} | N/A |")

    if best_field is not None and best_value is not None:
        best_label, best_surface = _GLOBAL_STRESS_LABELS[best_field]
        lines.append("")
        lines.append(
            f"**Overall {agg_label}:** {best_value:.6g} MPa "
            f"at **{best_label}** ({best_surface} surface)"
        )

    return "\n".join(lines)


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
    fallback_path = Path(default_vtk_file).resolve() if default_vtk_file else None
    plot_urls = [url for url in (available_plot_urls or []) if isinstance(url, str) and url.strip()]

    intent: Optional[LookupIntent] = _parse_agentic_intent(
        user_input,
        llm=llm,
        default_vtk_file=fallback_path,
        allow_implicit=allow_implicit,
    )

    if intent is None:
        return None

    # Qualitative questions (analysis, interpretation, inferences) should fall through
    # to generate_vtk_agent_response rather than returning a deterministic field listing.
    if intent.action == "detail":
        lowered_input = " ".join((user_input or "").strip().lower().split())
        qualitative_terms = {
            "analyse", "analyze", "explain", "interpret", "inference", "infer",
            "describe", "tell me about", "discuss", "insights", "what can you",
            "observations", "understand", "give me insight", "give me inference",
            "look at the", "examine", "what does it", "what do you see",
            "what can we", "read the plot", "read the graph", "read the result",
        }
        listing_terms = {
            "list fields", "list vtk", "what fields", "available fields",
            "show fields", "vtk info", "vtk summary", "vtk details", "list all fields",
        }
        has_qualitative = any(term in lowered_input for term in qualitative_terms)
        has_listing = any(term in lowered_input for term in listing_terms)
        if has_qualitative and not has_listing:
            return None

    can_narrate = bool(narration_url and narration_model)
    local_plots = _resolve_plot_paths(plot_urls) if can_narrate else []

    if intent.action == "plots":
        if can_narrate and local_plots:
            narrated = _narrate_plots_with_kimi(
                ollama_url=narration_url,
                model=narration_model,
                api_key=narration_api_key,
                user_question=user_input,
                local_plot_paths=local_plots,
                plot_urls=plot_urls,
            )
            if narrated:
                return narrated
        return _format_plot_gallery(plot_urls)

    if not intent.vtk_file.exists():
        return (
            "The analysis results are not available yet. "
            "Please run the analysis first."
        )

    try:
        if intent.action == "flexural":
            args = ["--file", str(intent.vtk_file)]
            # Plots are always generated by the script; no --plot flag needed.
            payload = _run_script(FLEXURAL_SCRIPT, args)
            formatted = _format_flexural_response(payload, api_base_url=api_base_url)
            if can_narrate:
                flexural_plots = _resolve_plot_paths(
                    [f"{api_base_url}/artifacts/{Path(p).name}" for p in (payload.get("contour_plots") or [])]
                ) + local_plots
                narrated = _narrate_with_kimi(
                    ollama_url=narration_url,
                    model=narration_model,
                    api_key=narration_api_key,
                    user_question=user_input,
                    structured_data=formatted,
                    local_plot_paths=flexural_plots or local_plots,
                )
                if narrated:
                    return narrated
            return formatted

        if intent.action == "detail":
            payload = _run_script(DETAIL_SCRIPT, ["--file", str(intent.vtk_file)])
            formatted = _format_detail_response(payload)
            if can_narrate:
                narrated = _narrate_with_kimi(
                    ollama_url=narration_url,
                    model=narration_model,
                    api_key=narration_api_key,
                    user_question=user_input,
                    structured_data=formatted,
                    local_plot_paths=local_plots,
                )
                if narrated:
                    return narrated
            return formatted

        if intent.action == "aggregate":
            # When the field was not explicitly specified and the query involves generic stress,
            # run a global multi-field scan across all 4 flexural stress fields.
            lowered_input = " ".join((user_input or "").strip().lower().split())
            if not intent.field_explicit and any(t in lowered_input for t in {"stress", "sigma"}):
                formatted = _run_global_stress_query(intent.vtk_file, intent.aggregation or "max")
                if can_narrate:
                    narrated = _narrate_with_kimi(
                        ollama_url=narration_url,
                        model=narration_model,
                        api_key=narration_api_key,
                        user_question=user_input,
                        structured_data=formatted,
                        local_plot_paths=local_plots,
                    )
                    if narrated:
                        return narrated
                return formatted

            args = [
                "--file", str(intent.vtk_file),
                "--field", str(intent.field),
                "--agg", str(intent.aggregation),
            ]
            if intent.component:
                args.extend(["--component", intent.component])
            if intent.use_abs:
                args.append("--abs")
            payload = _run_script(AGG_SCRIPT, args)
            formatted = _format_aggregate_response(payload)
            if can_narrate:
                narrated = _narrate_with_kimi(
                    ollama_url=narration_url,
                    model=narration_model,
                    api_key=narration_api_key,
                    user_question=user_input,
                    structured_data=formatted,
                    local_plot_paths=local_plots,
                )
                if narrated:
                    return narrated
            return formatted

        return "I understood this as a VTK lookup request, but could not determine the action."
    except Exception as exc:
        return f"VTK lookup failed: {exc}"
