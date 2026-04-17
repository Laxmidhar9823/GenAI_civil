from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VTK_FILE = REPO_ROOT / "output_python_square" / "results.vtk"
DETAIL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_detailed_report.py"
AGG_SCRIPT = REPO_ROOT / "backend" / "scripts" / "vtk_aggregate_query.py"


@dataclass
class LookupIntent:
    action: str
    vtk_file: Path
    aggregation: Optional[str] = None
    field: Optional[str] = None
    component: Optional[str] = None
    use_abs: bool = False


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


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


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

    return LookupIntent(action="detail", vtk_file=vtk_file)


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


def _format_detail_response(payload: Dict) -> str:
    point_fields = payload.get("available_fields", {}).get("point_data", [])
    cell_fields = payload.get("available_fields", {}).get("cell_data", [])

    lines = [
        "### VTK Detailed Report",
        f"- File: {payload.get('file', 'unknown')}",
        f"- Format: {payload.get('source_format', 'unknown')}",
        f"- Points: {payload.get('num_points', 0)}",
        f"- Cells: {payload.get('num_cells', 0)}",
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

    lines = [
        "### VTK Lookup Result",
        f"- File: {payload.get('file', 'unknown')}",
        f"- Field: {payload.get('resolved_field', payload.get('requested_field', 'unknown'))}",
        f"- Location: {payload.get('location', 'unknown')}",
        f"- Component: {payload.get('component', 'scalar')}",
        f"- Aggregation: {payload.get('aggregation', 'unknown')}",
        f"- Sample size: {payload.get('sample_size', 'unknown')}",
        f"- Value: {value_text}",
    ]
    return "\n".join(lines)


def handle_vtk_lookup_command(
    user_input: str,
    *,
    default_vtk_file: Optional[str] = None,
    allow_implicit: bool = False,
) -> Optional[str]:
    fallback_path = Path(default_vtk_file).resolve() if default_vtk_file else None
    intent = _parse_intent(
        user_input,
        default_vtk_file=fallback_path,
        allow_implicit=allow_implicit,
    )
    if intent is None:
        return None

    if not intent.vtk_file.exists():
        return (
            "I could not find the VTK file for lookup. "
            f"Expected: {intent.vtk_file}. "
            "Run the solver first, or provide an explicit .vtk path in your message."
        )

    try:
        if intent.action == "detail":
            payload = _run_script(
                DETAIL_SCRIPT,
                ["--file", str(intent.vtk_file)],
            )
            return _format_detail_response(payload)

        if intent.action == "aggregate":
            args = [
                "--file",
                str(intent.vtk_file),
                "--field",
                str(intent.field),
                "--agg",
                str(intent.aggregation),
            ]
            if intent.component:
                args.extend(["--component", intent.component])
            if intent.use_abs:
                args.append("--abs")

            payload = _run_script(AGG_SCRIPT, args)
            return _format_aggregate_response(payload)

        return "I understood this as a VTK lookup request, but could not determine the action."
    except Exception as exc:
        return f"VTK lookup failed: {exc}"
