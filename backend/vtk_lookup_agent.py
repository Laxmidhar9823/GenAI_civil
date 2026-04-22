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


_FIELD_ALIASES: Dict[str, List[str]] = {
    "w": ["w", "deflection", "vertical displacement", "vertical deflection", "out of plane displacement"],
    "u": ["u", "x displacement", "displacement x", "longitudinal displacement"],
    "v": ["v", "y displacement", "displacement y", "transverse displacement"],
    "theta_x": ["theta_x", "rotation x", "rotation about y", "slope x"],
    "theta_y": ["theta_y", "rotation y", "rotation about x", "slope y"],
    "sxx_top": ["sxx top", "sigma xx top", "top sigma xx", "top sxx", "top normal stress x"],
    "syy_top": ["syy top", "sigma yy top", "top sigma yy", "top syy"],
    "sxy_top": ["sxy top", "tau xy top", "top shear stress", "top sxy"],
    "sxx_bottom": ["sxx bottom", "sigma xx bottom", "bottom sigma xx", "bottom sxx", "sxx bot"],
    "syy_bottom": ["syy bottom", "sigma yy bottom", "bottom sigma yy", "bottom syy", "syy bot"],
    "sxy_bottom": ["sxy bottom", "tau xy bottom", "bottom shear stress", "bottom sxy", "sxy bot"],
    "sxx_membrane": ["sxx membrane", "sigma xx membrane", "midplane sigma xx", "membrane sxx"],
    "syy_membrane": ["syy membrane", "sigma yy membrane", "midplane sigma yy", "membrane syy"],
    "sxy_membrane": ["sxy membrane", "tau xy membrane", "midplane shear stress", "membrane sxy"],
    "displacement": ["displacement vector", "vector displacement", "displacement"],
}

_AGG_KEYWORDS: Dict[str, str] = {
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "maximum": "max",
    "highest": "max",
    "peak": "max",
    "max": "max",
    "minimum": "min",
    "lowest": "min",
    "min": "min",
    "sum": "sum",
    "total": "sum",
    "std": "std",
    "stdev": "std",
    "standard deviation": "std",
    "variance": "var",
    "var": "var",
    "median": "median",
    "count": "count",
    "p95": "p95",
    "95 percentile": "p95",
    "percentile 95": "p95",
    "p05": "p05",
    "p5": "p05",
    "5 percentile": "p05",
    "percentile 5": "p05",
}

_DETAIL_HINTS = (
    "analyze vtk",
    "analyse vtk",
    "inspect vtk",
    "vtk details",
    "vtk summary",
    "list vtk fields",
    "what fields",
    "show vtk info",
)

_FLEXURAL_HINTS = (
    "flexural stress",
    "bending stress",
    "sigma_x",
    "sigma_y",
    "stress report",
    "stress analysis",
    "top surface stress",
    "bottom surface stress",
    "stress contour",
    "contour plot",
)

_IMPLICIT_RESULT_HINTS = (
    "deflection",
    "displacement",
    "rotation",
    "stress",
    "sxx",
    "syy",
    "sxy",
    "result",
    "field",
    "plot",
    "graph",
)

_AGG_CANONICAL = {"mean", "max", "min", "sum", "std", "var", "median", "count", "p95", "p05"}

_PLOT_REQUEST_HINTS = (
    "show plot",
    "show plots",
    "generated plots",
    "view plots",
    "plot image",
    "plot images",
    "show contour",
    "show contours",
    "display plots",
)

# Questions asking for qualitative interpretation should bypass deterministic lookup
# and be handled by the intelligent VTK agent (generate_vtk_agent_response).
_QUALITATIVE_ANALYSIS_TERMS = frozenset({
    "analyse", "analyze", "explain", "interpret", "inference", "infer",
    "describe", "tell me about", "discuss", "insights", "what can you",
    "observations", "understand", "give me insight", "give me inference",
    "look at the", "examine", "what does it", "what do you see",
    "what can we", "read the plot", "read the graph", "read the result",
})

_FIELD_LISTING_TERMS = frozenset({
    "list fields", "list vtk", "what fields", "available fields",
    "show fields", "vtk info", "vtk summary", "vtk details", "list all fields",
})


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


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


def _detect_aggregation(text: str) -> Optional[str]:
    lowered = _normalize_text(text)
    for keyword, agg in _AGG_KEYWORDS.items():
        if keyword in lowered:
            return agg
    return None


def _detect_component(text: str) -> Optional[str]:
    lowered = _normalize_text(text)
    if "magnitude" in lowered or "norm" in lowered:
        return "magnitude"
    if re.search(r"\bcomponent\s*0\b|\bx\s*component\b", lowered):
        return "x"
    if re.search(r"\bcomponent\s*1\b|\by\s*component\b", lowered):
        return "y"
    if re.search(r"\bcomponent\s*2\b|\bz\s*component\b", lowered):
        return "z"
    if re.search(r"\bx\b", lowered) and "displacement" in lowered:
        return "x"
    if re.search(r"\by\b", lowered) and "displacement" in lowered:
        return "y"
    if re.search(r"\bz\b", lowered) and "displacement" in lowered:
        return "z"
    return None


def _detect_field(text: str) -> Optional[str]:
    lowered = _normalize_text(text)

    # Prefer exact known aliases.
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                return canonical

    # If user gave an explicit solver-style field token, preserve it.
    explicit = re.search(r"\b([a-z]+_[a-z]+|[usvwt]{1}|sxx|syy|sxy)\b", lowered)
    if explicit:
        token = explicit.group(1)
        return token

    return None


def _coerce_aggregation(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = _normalize_text(str(value))
    if not normalized:
        return None
    if normalized in _AGG_CANONICAL:
        return normalized
    return _AGG_KEYWORDS.get(normalized)


def _is_plot_request(text: str) -> bool:
    lowered = _normalize_text(text)
    return any(hint in lowered for hint in _PLOT_REQUEST_HINTS)


def _is_lookup_intent(text: str, allow_implicit: bool = False) -> bool:
    lowered = _normalize_text(text)
    has_vtk_ref = (
        "vtk" in lowered
        or bool(re.search(r"\.(?:vtk|vtu|vtp|vtr|vts|vti)\b", lowered))
        or "result file" in lowered
    )

    if not has_vtk_ref and not allow_implicit:
        return False

    if not has_vtk_ref and allow_implicit:
        has_result_hint = any(h in lowered for h in _IMPLICIT_RESULT_HINTS)
        if has_result_hint and (_detect_aggregation(lowered) is not None or _detect_field(lowered) is not None):
            return True

        if has_result_hint and any(phrase in lowered for phrase in ("what is", "show", "compare", "give me", "how much")):
            return True

        return False

    if any(hint in lowered for hint in _DETAIL_HINTS):
        return True

    if any(hint in lowered for hint in _FLEXURAL_HINTS):
        return True

    if _detect_aggregation(lowered):
        return True

    if "lookup" in lowered or "query" in lowered:
        return True

    return False


def _parse_intent(
    text: str,
    *,
    default_vtk_file: Optional[Path] = None,
    allow_implicit: bool = False,
) -> Optional[LookupIntent]:
    if not _is_lookup_intent(text, allow_implicit=allow_implicit):
        return None

    lowered = _normalize_text(text)
    vtk_file = _extract_vtk_path(text, default_vtk_file=default_vtk_file)
    agg = _detect_aggregation(lowered)
    field = _detect_field(lowered)
    component = _detect_component(lowered)
    use_abs = "absolute" in lowered or "abs" in lowered

    if any(hint in lowered for hint in _FLEXURAL_HINTS):
        with_plot = "contour" in lowered or "plot" in lowered
        return LookupIntent(action="flexural", vtk_file=vtk_file, with_plot=with_plot)

    if agg:
        if not field:
            # Sensible default when user asks generic aggregation on displacement/results.
            field = "w"
        return LookupIntent(
            action="aggregate",
            vtk_file=vtk_file,
            aggregation=agg,
            field=field,
            component=component,
            use_abs=use_abs,
        )

    if _is_plot_request(lowered):
        return LookupIntent(action="plots", vtk_file=vtk_file)

    return LookupIntent(action="detail", vtk_file=vtk_file)


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
        available_fields = sorted(_FIELD_ALIASES.keys())

    return (
        "Classify and route the user query into one VTK action.\n"
        "Return STRICT JSON with keys:\n"
        "{\n"
        '  "is_vtk_query": boolean,\n'
        '  "action": "detail" | "aggregate" | "flexural" | "plots" | "none",\n'
        '  "field": string | null,\n'
        '  "aggregation": "mean"|"max"|"min"|"sum"|"std"|"var"|"median"|"count"|"p95"|"p05"|null,\n'
        '  "component": "x"|"y"|"z"|"magnitude"|null,\n'
        '  "use_abs": boolean,\n'
        '  "with_plot": boolean\n'
        "}\n\n"
        "Routing rules:\n"
        "- detail: asks what fields/summary/info/list.\n"
        "- aggregate: asks numeric statistic (mean/max/min/etc.) over one field.\n"
        "- flexural: asks bending/flexural/top/bottom stress report.\n"
        "- plots: asks to show/view generated plots or contour images.\n"
        "- none: unrelated to VTK results.\n"
        "- For aggregate, if no field is explicit but user asks deflection/displacement, use field='w'.\n"
        "- Prefer field names from AVAILABLE_FIELDS.\n"
        "- If the message is unrelated and allow_implicit is false, choose none.\n\n"
        f"allow_implicit={str(bool(allow_implicit)).lower()}\n"
        f"vtk_file={vtk_file}\n"
        f"AVAILABLE_FIELDS={', '.join(available_fields)}\n"
        f"USER_QUERY={user_input}"
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
    system = SystemMessage(
        content=(
            "You are a VTK tool router for pavement analysis. "
            "Return JSON only. Never explain your reasoning."
        )
    )
    human = HumanMessage(content=_build_planner_prompt(user_input=user_input, vtk_file=vtk_file, allow_implicit=allow_implicit))

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

    action = _normalize_text(str(payload.get("action") or "none"))
    if action in {"", "none", "null"}:
        return None

    if action == "plots":
        return LookupIntent(action="plots", vtk_file=vtk_file)

    if action == "detail":
        return LookupIntent(action="detail", vtk_file=vtk_file)

    if action == "flexural":
        with_plot = bool(payload.get("with_plot")) or _is_plot_request(user_input)
        return LookupIntent(action="flexural", vtk_file=vtk_file, with_plot=with_plot)

    if action == "aggregate":
        requested_field = payload.get("field")
        requested_component = payload.get("component")
        requested_agg = payload.get("aggregation")

        field = _detect_field(str(requested_field)) if requested_field else None
        if not field:
            field = _detect_field(user_input)
        if not field:
            field = "w"

        aggregation = _coerce_aggregation(str(requested_agg) if requested_agg is not None else None)
        if not aggregation:
            aggregation = _detect_aggregation(user_input) or "mean"

        component = None
        if isinstance(requested_component, str):
            normalized_component = _normalize_text(requested_component)
            if normalized_component in {"x", "y", "z", "magnitude"}:
                component = normalized_component
        if component is None:
            component = _detect_component(user_input)

        use_abs = bool(payload.get("use_abs")) or ("absolute" in _normalize_text(user_input))
        return LookupIntent(
            action="aggregate",
            vtk_file=vtk_file,
            aggregation=aggregation,
            field=field,
            component=component,
            use_abs=use_abs,
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
        "| Field | Surface | Max (MPa) | Min (MPa) | Mean (MPa) |",
        "|-------|---------|-----------|-----------|------------|",
    ]

    field_order = ["sxx_top", "sxx_bottom", "syy_top", "syy_bottom"]
    labels = {
        "sxx_top": "σ_x",
        "sxx_bottom": "σ_x",
        "syy_top": "σ_y",
        "syy_bottom": "σ_y",
    }

    for field_name in field_order:
        r = results.get(field_name)
        if r is None:
            continue
        if not r.get("available", False):
            lines.append(f"| {labels[field_name]} | {r['surface']} | N/A | N/A | N/A |")
            continue
        mx = f"{r['max']:.6g}" if r["max"] is not None else "N/A"
        mn = f"{r['min']:.6g}" if r["min"] is not None else "N/A"
        me = f"{r['mean']:.6g}" if r["mean"] is not None else "N/A"
        lines.append(f"| {labels[field_name]} | {r['surface']} | {mx} | {mn} | {me} |")

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
    fallback_path = Path(default_vtk_file).resolve() if default_vtk_file else None
    plot_urls = [url for url in (available_plot_urls or []) if isinstance(url, str) and url.strip()]

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

    if intent is None:
        return None

    # Qualitative questions (analysis, interpretation, inferences) should fall through
    # to generate_vtk_agent_response rather than returning a deterministic field listing.
    if intent.action == "detail":
        lowered_input = _normalize_text(user_input)
        has_qualitative = any(term in lowered_input for term in _QUALITATIVE_ANALYSIS_TERMS)
        has_listing = any(term in lowered_input for term in _FIELD_LISTING_TERMS)
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
            if intent.with_plot:
                args.append("--plot")
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
