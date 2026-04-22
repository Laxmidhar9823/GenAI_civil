from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class VTKReadError(RuntimeError):
    """Raised when a VTK file cannot be parsed."""


def _ensure_numeric_array(array: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(array)
    except Exception:
        return None

    if arr.dtype.kind not in {"i", "u", "f", "b"}:
        return None

    return arr.astype(float, copy=False)


def _read_float_slice(tokens: List[str], start: int, count: int) -> Tuple[np.ndarray, int]:
    end = start + count
    if end > len(tokens):
        raise VTKReadError("Unexpected end of VTK file while reading numeric values.")
    try:
        values = np.array([float(tok) for tok in tokens[start:end]], dtype=float)
    except ValueError as exc:
        raise VTKReadError("VTK file contains invalid numeric values.") from exc
    return values, end


def _find_token(tokens: List[str], token: str, start: int = 0) -> int:
    token_u = token.upper()
    for idx in range(start, len(tokens)):
        if tokens[idx].upper() == token_u:
            return idx
    return -1


def _read_legacy_ascii_vtk(file_path: Path) -> Dict[str, Any]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    tokens = text.split()
    if not tokens:
        raise VTKReadError("Empty VTK file.")

    dataset_idx = _find_token(tokens, "DATASET")
    if dataset_idx == -1 or dataset_idx + 1 >= len(tokens):
        raise VTKReadError("DATASET section is missing in VTK file.")

    dataset_type = tokens[dataset_idx + 1].upper()
    if dataset_type != "UNSTRUCTURED_GRID":
        raise VTKReadError(
            f"Unsupported legacy VTK dataset type '{dataset_type}'. "
            "Only UNSTRUCTURED_GRID is supported without meshio."
        )

    points_idx = _find_token(tokens, "POINTS", dataset_idx)
    if points_idx == -1:
        raise VTKReadError("POINTS section is missing in VTK file.")

    try:
        num_points = int(tokens[points_idx + 1])
    except Exception as exc:
        raise VTKReadError("Invalid POINTS section in VTK file.") from exc

    pos = points_idx + 3
    points_flat, pos = _read_float_slice(tokens, pos, 3 * num_points)
    points = points_flat.reshape(num_points, 3)

    cells_idx = _find_token(tokens, "CELLS", pos)
    if cells_idx == -1:
        raise VTKReadError("CELLS section is missing in VTK file.")

    try:
        num_cells = int(tokens[cells_idx + 1])
    except Exception as exc:
        raise VTKReadError("Invalid CELLS section in VTK file.") from exc

    pos = cells_idx + 3
    cells: List[List[int]] = []
    for _ in range(num_cells):
        if pos >= len(tokens):
            raise VTKReadError("Unexpected end of VTK file while reading cells.")
        try:
            count = int(tokens[pos])
        except ValueError as exc:
            raise VTKReadError("Invalid cell size in CELLS section.") from exc
        pos += 1
        if pos + count > len(tokens):
            raise VTKReadError("Unexpected end of VTK file while reading cell connectivity.")
        try:
            cells.append([int(tok) for tok in tokens[pos : pos + count]])
        except ValueError as exc:
            raise VTKReadError("Invalid cell connectivity value in CELLS section.") from exc
        pos += count

    cell_types_idx = _find_token(tokens, "CELL_TYPES", pos)
    cell_types: np.ndarray
    if cell_types_idx != -1:
        try:
            num_cell_types = int(tokens[cell_types_idx + 1])
        except Exception as exc:
            raise VTKReadError("Invalid CELL_TYPES section in VTK file.") from exc
        pos = cell_types_idx + 2
        cell_types_flat, pos = _read_float_slice(tokens, pos, num_cell_types)
        cell_types = cell_types_flat.astype(int)
    else:
        cell_types = np.array([], dtype=int)

    point_data: Dict[str, np.ndarray] = {}
    cell_data: Dict[str, np.ndarray] = {}

    current_location: Optional[str] = None
    expected_count = 0

    while pos < len(tokens):
        token = tokens[pos].upper()

        if token == "POINT_DATA":
            if pos + 1 >= len(tokens):
                raise VTKReadError("POINT_DATA section is malformed.")
            current_location = "point_data"
            expected_count = int(tokens[pos + 1])
            pos += 2
            continue

        if token == "CELL_DATA":
            if pos + 1 >= len(tokens):
                raise VTKReadError("CELL_DATA section is malformed.")
            current_location = "cell_data"
            expected_count = int(tokens[pos + 1])
            pos += 2
            continue

        if token == "SCALARS":
            if current_location is None:
                raise VTKReadError("SCALARS encountered before POINT_DATA/CELL_DATA section.")
            if pos + 2 >= len(tokens):
                raise VTKReadError("SCALARS declaration is malformed.")

            name = tokens[pos + 1]
            num_components = 1
            if pos + 3 < len(tokens):
                try:
                    num_components = int(tokens[pos + 3])
                    pos += 4
                except ValueError:
                    pos += 3
            else:
                pos += 3

            if pos + 1 < len(tokens) and tokens[pos].upper() == "LOOKUP_TABLE":
                pos += 2

            values, pos = _read_float_slice(tokens, pos, expected_count * num_components)
            data = values.reshape(expected_count, num_components) if num_components > 1 else values

            if current_location == "point_data":
                point_data[name] = data
            else:
                cell_data[name] = data
            continue

        if token == "VECTORS":
            if current_location is None:
                raise VTKReadError("VECTORS encountered before POINT_DATA/CELL_DATA section.")
            if pos + 2 >= len(tokens):
                raise VTKReadError("VECTORS declaration is malformed.")
            name = tokens[pos + 1]
            pos += 3
            values, pos = _read_float_slice(tokens, pos, expected_count * 3)
            data = values.reshape(expected_count, 3)
            if current_location == "point_data":
                point_data[name] = data
            else:
                cell_data[name] = data
            continue

        if token == "TENSORS":
            if current_location is None:
                raise VTKReadError("TENSORS encountered before POINT_DATA/CELL_DATA section.")
            if pos + 2 >= len(tokens):
                raise VTKReadError("TENSORS declaration is malformed.")
            name = tokens[pos + 1]
            pos += 3
            values, pos = _read_float_slice(tokens, pos, expected_count * 9)
            data = values.reshape(expected_count, 9)
            if current_location == "point_data":
                point_data[name] = data
            else:
                cell_data[name] = data
            continue

        if token == "FIELD":
            if current_location is None:
                raise VTKReadError("FIELD encountered before POINT_DATA/CELL_DATA section.")
            if pos + 2 >= len(tokens):
                raise VTKReadError("FIELD declaration is malformed.")
            try:
                num_arrays = int(tokens[pos + 2])
            except ValueError as exc:
                raise VTKReadError("Invalid FIELD array count.") from exc
            pos += 3
            for _ in range(num_arrays):
                if pos + 3 >= len(tokens):
                    raise VTKReadError("FIELD array declaration is malformed.")
                name = tokens[pos]
                num_components = int(tokens[pos + 1])
                num_tuples = int(tokens[pos + 2])
                pos += 4
                values, pos = _read_float_slice(tokens, pos, num_components * num_tuples)
                data = values.reshape(num_tuples, num_components) if num_components > 1 else values
                if current_location == "point_data":
                    point_data[name] = data
                else:
                    cell_data[name] = data
            continue

        pos += 1

    return {
        "source": str(file_path),
        "source_format": "legacy-ascii-vtk",
        "points": points,
        "cells": cells,
        "cell_types": cell_types,
        "point_data": point_data,
        "cell_data": cell_data,
    }


def load_vtk_dataset(file_path: str) -> Dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"VTK file not found: {path}")

    meshio_error: Optional[Exception] = None
    try:
        import meshio  # type: ignore

        mesh = meshio.read(str(path))
        point_data: Dict[str, np.ndarray] = {}
        for name, values in mesh.point_data.items():
            arr = _ensure_numeric_array(values)
            if arr is not None:
                point_data[name] = arr

        cell_data: Dict[str, np.ndarray] = {}
        for name, block_values in mesh.cell_data.items():
            numeric_blocks: List[np.ndarray] = []
            for block in block_values:
                arr = _ensure_numeric_array(block)
                if arr is not None:
                    numeric_blocks.append(arr)
            if not numeric_blocks:
                continue
            if len(numeric_blocks) == 1:
                cell_data[name] = numeric_blocks[0]
            else:
                try:
                    cell_data[name] = np.concatenate(numeric_blocks, axis=0)
                except Exception:
                    # Keep first compatible block if shapes differ.
                    cell_data[name] = numeric_blocks[0]

        cells: List[List[int]] = []
        for cell_block in mesh.cells:
            for row in np.asarray(cell_block.data):
                cells.append([int(idx) for idx in row])

        return {
            "source": str(path),
            "source_format": f"meshio:{path.suffix.lower() or 'unknown'}",
            "points": np.asarray(mesh.points, dtype=float),
            "cells": cells,
            "cell_types": np.array([], dtype=int),
            "point_data": point_data,
            "cell_data": cell_data,
        }
    except Exception as exc:
        meshio_error = exc

    if path.suffix.lower() == ".vtk":
        try:
            return _read_legacy_ascii_vtk(path)
        except Exception as exc:
            if meshio_error is not None:
                raise VTKReadError(
                    f"Failed to read VTK via meshio ({meshio_error}) and legacy parser ({exc})."
                ) from exc
            raise

    if meshio_error is not None:
        raise VTKReadError(
            "Unable to parse VTK file. Install meshio for this file format, or provide an ASCII legacy .vtk file. "
            f"Original error: {meshio_error}"
        ) from meshio_error

    raise VTKReadError("Unable to parse the provided VTK file.")


def _flatten_components(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if data.ndim == 1:
        return data.reshape(-1, 1)
    first_dim = data.shape[0]
    return data.reshape(first_dim, -1)


def _safe_stat_values(values: np.ndarray) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "sum": None,
            "p05": None,
            "p95": None,
        }

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "sum": float(np.sum(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def summarize_array(arr: np.ndarray) -> Dict[str, Any]:
    components = _flatten_components(np.asarray(arr, dtype=float))
    num_tuples = int(components.shape[0])
    num_components = int(components.shape[1])

    finite_mask = np.all(np.isfinite(components), axis=1)
    finite_components = components[finite_mask]

    component_stats: List[Dict[str, Any]] = []
    for idx in range(num_components):
        values = finite_components[:, idx] if finite_components.size else np.array([], dtype=float)
        comp_name = "scalar"
        if num_components > 1:
            comp_name = ["x", "y", "z"][idx] if idx < 3 else f"c{idx}"
        stats = _safe_stat_values(values)
        stats["component"] = comp_name
        component_stats.append(stats)

    summary: Dict[str, Any] = {
        "shape": list(np.asarray(arr).shape),
        "num_tuples": num_tuples,
        "num_components": num_components,
        "finite_tuples": int(np.sum(finite_mask)),
        "non_finite_tuples": int(num_tuples - np.sum(finite_mask)),
        "component_stats": component_stats,
    }

    if num_components > 1:
        magnitude = np.linalg.norm(finite_components, axis=1) if finite_components.size else np.array([], dtype=float)
        summary["magnitude_stats"] = _safe_stat_values(magnitude)

    return summary


def describe_dataset(file_path: str) -> Dict[str, Any]:
    dataset = load_vtk_dataset(file_path)

    point_fields: List[Dict[str, Any]] = []
    for name, arr in dataset["point_data"].items():
        point_fields.append({"name": name, **summarize_array(arr)})

    cell_fields: List[Dict[str, Any]] = []
    for name, arr in dataset["cell_data"].items():
        cell_fields.append({"name": name, **summarize_array(arr)})

    return {
        "file": dataset["source"],
        "source_format": dataset["source_format"],
        "num_points": int(np.asarray(dataset["points"]).shape[0]),
        "num_cells": int(len(dataset["cells"])),
        "cell_types": [int(v) for v in np.asarray(dataset.get("cell_types", np.array([], dtype=int))).tolist()],
        "available_fields": {
            "point_data": [f["name"] for f in point_fields],
            "cell_data": [f["name"] for f in cell_fields],
        },
        "point_data_fields": point_fields,
        "cell_data_fields": cell_fields,
    }


_FLEXURAL_FIELDS = {
    "sxx_top": ("sigma_x", "top"),
    "syy_top": ("sigma_y", "top"),
    "sxx_bottom": ("sigma_x", "bottom"),
    "syy_bottom": ("sigma_y", "bottom"),
}


def flexural_stress_analysis(file_path: str) -> Dict[str, Any]:
    """Return max, min, and mean for sigma_x and sigma_y at top and bottom surfaces."""
    dataset = load_vtk_dataset(file_path)
    point_data = dataset["point_data"]

    results: Dict[str, Any] = {}
    for field_name, (stress_label, surface) in _FLEXURAL_FIELDS.items():
        if field_name not in point_data:
            results[field_name] = {
                "stress": stress_label,
                "surface": surface,
                "available": False,
            }
            continue

        arr = np.asarray(point_data[field_name], dtype=float).ravel()
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            results[field_name] = {
                "stress": stress_label,
                "surface": surface,
                "available": True,
                "max": None,
                "min": None,
                "mean": None,
                "sample_size": 0,
            }
        else:
            results[field_name] = {
                "stress": stress_label,
                "surface": surface,
                "available": True,
                "max": float(np.max(finite)),
                "min": float(np.min(finite)),
                "mean": float(np.mean(finite)),
                "sample_size": int(finite.size),
            }

    missing = [f for f, v in results.items() if not v.get("available", False)]
    return {
        "file": dataset["source"],
        "source_format": dataset["source_format"],
        "results": results,
        "missing_fields": missing,
    }


def plot_flexural_contours(
    file_path: str,
    output_dir: Optional[str] = None,
) -> List[str]:
    """Generate contour plots for sigma_x and sigma_y at top and bottom surfaces.

    Returns a list of saved PNG file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    dataset = load_vtk_dataset(file_path)
    points = np.asarray(dataset["points"], dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    point_data = dataset["point_data"]

    if output_dir is None:
        out_path = Path(dataset["source"]).parent
    else:
        out_path = Path(output_dir)

    saved: List[str] = []

    plot_specs = [
        ("sxx_top", r"$\sigma_x$ — Top Surface", "sigma_x_top"),
        ("sxx_bottom", r"$\sigma_x$ — Bottom Surface", "sigma_x_bottom"),
        ("syy_top", r"$\sigma_y$ — Top Surface", "sigma_y_top"),
        ("syy_bottom", r"$\sigma_y$ — Bottom Surface", "sigma_y_bottom"),
    ]

    triangulation = mtri.Triangulation(x, y)

    for field_name, title, file_stem in plot_specs:
        if field_name not in point_data:
            continue

        values = np.asarray(point_data[field_name], dtype=float).ravel()
        if values.size != x.size:
            continue

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue

        abs_max = float(np.max(np.abs(values[finite_mask])))
        if abs_max == 0:
            abs_max = 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        tcf = ax.tricontourf(
            triangulation,
            values,
            levels=20,
            cmap="RdBu_r",
            vmin=-abs_max,
            vmax=abs_max,
        )
        ax.tricontour(
            triangulation,
            values,
            levels=10,
            colors="k",
            linewidths=0.3,
            alpha=0.4,
        )
        cbar = fig.colorbar(tcf, ax=ax)
        cbar.set_label("Stress (MPa)", fontsize=11)
        ax.set_xlabel("x (mm)", fontsize=11)
        ax.set_ylabel("y (mm)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        fig.tight_layout()

        png_path = out_path / f"{file_stem}.png"
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)
        saved.append(str(png_path))

    return saved


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


def _aggregation_alias(agg: str) -> str:
    norm = _normalize_key(agg)
    mapping = {
        "avg": "mean",
        "average": "mean",
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
        "standarddeviation": "std",
        "variance": "var",
        "var": "var",
        "median": "median",
        "count": "count",
        "p95": "p95",
        "95percentile": "p95",
        "percentile95": "p95",
        "p05": "p05",
        "p5": "p05",
        "5percentile": "p05",
        "percentile5": "p05",
    }
    return mapping.get(norm, norm)


def _resolve_component(component: Optional[str], num_components: int) -> Tuple[Optional[int], str]:
    if num_components == 1:
        return 0, "scalar"

    if component is None:
        return None, "magnitude"

    text = (component or "").strip().lower()
    if text in {"magnitude", "mag", "norm", "abs"}:
        return None, "magnitude"

    if text in {"x", "0", "c0"}:
        return 0, "x"
    if text in {"y", "1", "c1"}:
        return 1, "y"
    if text in {"z", "2", "c2"}:
        return 2, "z"

    if text.startswith("c") and text[1:].isdigit():
        idx = int(text[1:])
        if 0 <= idx < num_components:
            return idx, f"c{idx}"

    if text.isdigit():
        idx = int(text)
        if 0 <= idx < num_components:
            label = ["x", "y", "z"][idx] if idx < 3 else f"c{idx}"
            return idx, label

    raise ValueError(
        f"Invalid component '{component}'. Use x/y/z, an index, or 'magnitude'."
    )


def _extract_field_values(
    array: np.ndarray,
    component: Optional[str],
    use_abs: bool,
) -> Tuple[np.ndarray, str]:
    components = _flatten_components(np.asarray(array, dtype=float))
    comp_index, comp_label = _resolve_component(component, components.shape[1])

    finite_rows = np.all(np.isfinite(components), axis=1)
    valid = components[finite_rows]
    if comp_index is None:
        values = np.linalg.norm(valid, axis=1)
    else:
        values = valid[:, comp_index]

    if use_abs:
        values = np.abs(values)

    return values, comp_label


def _resolve_field(dataset: Dict[str, Any], field: str, location: str) -> Tuple[str, str, np.ndarray]:
    point_data: Dict[str, np.ndarray] = dataset["point_data"]
    cell_data: Dict[str, np.ndarray] = dataset["cell_data"]

    def _match(candidates: Dict[str, np.ndarray], name: str) -> Optional[Tuple[str, np.ndarray]]:
        if name in candidates:
            return name, candidates[name]
        norm = _normalize_key(name)
        for key, arr in candidates.items():
            if _normalize_key(key) == norm:
                return key, arr
        return None

    wanted = (location or "auto").strip().lower()
    if wanted == "point":
        match = _match(point_data, field)
        if match:
            return "point_data", match[0], match[1]
        raise KeyError(f"Field '{field}' not found in point data.")

    if wanted == "cell":
        match = _match(cell_data, field)
        if match:
            return "cell_data", match[0], match[1]
        raise KeyError(f"Field '{field}' not found in cell data.")

    match = _match(point_data, field)
    if match:
        return "point_data", match[0], match[1]

    match = _match(cell_data, field)
    if match:
        return "cell_data", match[0], match[1]

    raise KeyError(f"Field '{field}' was not found in point or cell data.")


def aggregate_field(
    file_path: str,
    field: str,
    aggregation: str,
    location: str = "auto",
    component: Optional[str] = None,
    use_abs: bool = False,
) -> Dict[str, Any]:
    dataset = load_vtk_dataset(file_path)
    resolved_location, resolved_field, array = _resolve_field(dataset, field, location)

    values, component_used = _extract_field_values(array, component, use_abs)
    agg = _aggregation_alias(aggregation)

    if agg == "count":
        value = int(values.size)
    elif values.size == 0:
        raise ValueError("No finite values are available for the selected field/component.")
    elif agg == "mean":
        value = float(np.mean(values))
    elif agg == "max":
        value = float(np.max(values))
    elif agg == "min":
        value = float(np.min(values))
    elif agg == "sum":
        value = float(np.sum(values))
    elif agg == "std":
        value = float(np.std(values))
    elif agg == "var":
        value = float(np.var(values))
    elif agg == "median":
        value = float(np.median(values))
    elif agg == "p95":
        value = float(np.percentile(values, 95))
    elif agg == "p05":
        value = float(np.percentile(values, 5))
    else:
        raise ValueError(
            "Unsupported aggregation. Use one of: mean, max, min, sum, std, var, median, p95, p05, count."
        )

    return {
        "file": dataset["source"],
        "source_format": dataset["source_format"],
        "requested_field": field,
        "resolved_field": resolved_field,
        "location": resolved_location,
        "aggregation": agg,
        "component": component_used,
        "absolute": bool(use_abs),
        "sample_size": int(values.size),
        "value": value,
        "available_fields": {
            "point_data": list(dataset["point_data"].keys()),
            "cell_data": list(dataset["cell_data"].keys()),
        },
    }
