from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vtk_stats import VTKReadError, describe_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a detailed report for a VTK result file."
    )
    parser.add_argument("--file", required=True, help="Path to the VTK file (.vtk/.vtu/.vtp/...).")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    try:
        report = describe_dataset(args.file)
        if args.pretty:
            print(json.dumps(report, indent=2))
        else:
            print(json.dumps(report))
        return 0
    except (FileNotFoundError, VTKReadError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
