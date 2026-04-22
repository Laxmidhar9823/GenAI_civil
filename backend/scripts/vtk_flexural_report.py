from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.vtk_stats import VTKReadError, flexural_stress_analysis, plot_flexural_contours


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report flexural stresses (sigma_x, sigma_y) at top and bottom surfaces "
            "of a plate VTK result file. Outputs max, min, and mean for each "
            "(2 stresses × 2 surfaces × 3 statistics = 12 values) plus contour plots."
        )
    )
    parser.add_argument("--file", required=True, help="Path to the VTK result file.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Ignored — contour plots are always generated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs (default: same directory as VTK file).",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    try:
        report = flexural_stress_analysis(args.file)

        # Always generate contour plots for all 4 flexural stress fields and include paths.
        saved_plots = plot_flexural_contours(args.file, output_dir=args.output_dir)
        report["contour_plots"] = saved_plots

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
