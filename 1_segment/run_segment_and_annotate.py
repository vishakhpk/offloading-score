"""
Run both segmentation and annotation scripts in sequence.

This script:
1. Runs 0_segment.py to segment a processed trajectory
2. Runs 1_annotate.py to annotate the resulting segments
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEGMENT_SCRIPT = Path(__file__).parent / "0_segment.py"
ANNOTATE_SCRIPT = Path(__file__).parent / "1_annotate.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run segmentation and annotation in sequence."
    )
    
    # Required argument
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Path to processed_trajectory.json.",
    )
    
    # Segmentation arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="MSE threshold for splitting segments (overrides --auto-percentile).",
    )
    parser.add_argument(
        "--auto-percentile",
        type=float,
        default=75.0,
        help="Derive threshold from this percentile of scores (default: 75.0).",
    )
    parser.add_argument(
        "--do-remerge",
        action="store_true",
        help="Enable LLM-based re-merging of adjacent segments.",
    )
    parser.add_argument(
        "--remerge-sim-threshold",
        type=float,
        default=0.8,
        help="Minimum probability (0-1) that adjacent segments are same session to merge.",
    )
    
    # Annotation arguments
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run annotation even if an annotation already exists on the node.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=None,
        help="Minimum transition_diff to include screenshots in sequence annotations. Auto-detected from metadata if not provided.",
    )
    
    # Common arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip segmentation step (assumes segments already exist).",
    )
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skip annotation step (only run segmentation).",
    )
    
    return parser


def run_segmentation(args: argparse.Namespace) -> Path:
    """Run the segmentation script and return the output segments file path."""
    trajectory_path = args.trajectory.expanduser().resolve()
    
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
    
    # Build command for segmentation
    cmd = [sys.executable, str(SEGMENT_SCRIPT), "--trajectory", str(trajectory_path)]
    
    if args.threshold is not None:
        cmd.extend(["--threshold", str(args.threshold)])
    else:
        cmd.extend(["--auto-percentile", str(args.auto_percentile)])
    
    if args.do_remerge:
        cmd.append("--do-remerge")
        cmd.extend(["--remerge-sim-threshold", str(args.remerge_sim_threshold)])
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"[SEGMENT] Running segmentation on {trajectory_path}")
    print(f"[SEGMENT] Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True)
    
    # Determine output file path (using new organized structure)
    output_file = trajectory_path.parent / "1_segment" / "segments.json"
    
    if not output_file.exists():
        raise FileNotFoundError(
            f"Segmentation output file not found: {output_file}. "
            f"Segmentation may have failed."
        )
    
    print(f"[SEGMENT] Segmentation complete. Output: {output_file}")
    return output_file


def run_annotation(segments_file: Path, args: argparse.Namespace) -> None:
    """Run the annotation script on the segments file."""
    if not segments_file.exists():
        raise FileNotFoundError(f"Segments file not found: {segments_file}")
    
    # Build command for annotation
    cmd = [sys.executable, str(ANNOTATE_SCRIPT), "--input", str(segments_file)]
    
    if args.overwrite:
        cmd.append("--overwrite")
    
    if args.diff_threshold is not None:
        cmd.extend(["--diff-threshold", str(args.diff_threshold)])
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"\n[ANNOTATE] Running annotation on {segments_file}")
    print(f"[ANNOTATE] Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True)
    
    # Annotation script outputs to same directory as input (1_segment/)
    output_file = segments_file.parent / "annotated.json"
    if not output_file.exists():
        raise FileNotFoundError(
            f"Annotation output file not found: {output_file}. "
            f"Annotation may have failed."
        )
    print(f"[ANNOTATE] Annotation complete. Output: {output_file}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        # Step 1: Segmentation
        if not args.skip_segmentation:
            segments_file = run_segmentation(args)
        else:
            # Use expected output file name (using new organized structure)
            trajectory_path = args.trajectory.expanduser().resolve()
            segments_file = trajectory_path.parent / "1_segment" / "segments.json"
            if not segments_file.exists():
                raise FileNotFoundError(
                    f"Segments file not found: {segments_file}. "
                    f"Run segmentation first or remove --skip-segmentation flag."
                )
            print(f"[SKIP] Using existing segments file: {segments_file}")
        
        # Step 2: Annotation
        if not args.skip_annotation:
            run_annotation(segments_file, args)
        else:
            print("[SKIP] Skipping annotation step.")
        
        print("\n[COMPLETE] All steps finished successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Subprocess failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

