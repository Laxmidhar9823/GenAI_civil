from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vtk_stats import VTKReadError, aggregate_field


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query a VTK result file using common aggregations (mean/max/min/etc.)."
    )
    parser.add_argument("--file", required=True, help="Path to the VTK file (.vtk/.vtu/.vtp/...).")
    parser.add_argument("--field", required=True, help="Field name to aggregate.")
    parser.add_argument(
        "--agg",
        required=True,
        help="Aggregation: mean, max, min, sum, std, var, median, p95, p05, count.",
    )
    parser.add_argument(
        "--location",
        default="auto",
        choices=["auto", "point", "cell"],
        help="Where to search for the field (default: auto).",
    )
    parser.add_argument(
        "--component",
        default=None,
        help="Optional component for vector/tensor fields: x, y, z, index, or magnitude.",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        dest="use_abs",
        help="Aggregate absolute values.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    try:
        result = aggregate_field(
            file_path=args.file,
            field=args.field,
            aggregation=args.agg,
            location=args.location,
            component=args.component,
            use_abs=args.use_abs,
        )
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
        return 0
    except (FileNotFoundError, VTKReadError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
