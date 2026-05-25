"""
Run all timeline processing scripts in sequence.

This script:
1. Runs 0_time_segs.py to create timeline from labeled segments
2. Runs 1_close_gaps.py to close small gaps in timeline
3. Runs 2_add_pauses.py to add pauses from processed trajectory
4. Runs 3_classify_gaps.py to classify remaining gaps using LLM
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMELINE_SCRIPTS = {
    "time_segs": Path(__file__).parent / "0_time_segs.py",
    "close_gaps": Path(__file__).parent / "1_close_gaps.py",
    "add_pauses": Path(__file__).parent / "2_add_pauses.py",
    "classify_gaps": Path(__file__).parent / "3_classify_gaps.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all timeline processing scripts in sequence."
    )
    
    # Required arguments
    parser.add_argument(
        "--segments",
        type=Path,
        required=True,
        help="Path to labeled segments JSON file (input for 0_time_segs.py).",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="Path to processed_trajectory.json (input for 2_add_pauses.py).",
    )
    
    # Optional arguments for skipping steps
    parser.add_argument(
        "--skip-time-segs",
        action="store_true",
        help="Skip timeline creation step (use existing timeline file).",
    )
    parser.add_argument(
        "--skip-close-gaps",
        action="store_true",
        help="Skip gap closing step.",
    )
    parser.add_argument(
        "--skip-add-pauses",
        action="store_true",
        help="Skip pause addition step.",
    )
    parser.add_argument(
        "--skip-classify-gaps",
        action="store_true",
        help="Skip gap classification step.",
    )
    
    # Step 0: time_segs arguments
    parser.add_argument(
        "--timeline-output",
        type=Path,
        default=None,
        help="Output path for timeline JSON from 0_time_segs.py (default: segments path with _timeline suffix).",
    )
    parser.add_argument(
        "--display-timeline",
        action="store_true",
        help="Display human-readable timeline after creation.",
    )
    
    # Step 1: close_gaps arguments
    parser.add_argument(
        "--min-gap",
        type=float,
        default=5.0,
        help="Minimum gap duration in seconds to consider (default: 5.0).",
    )
    
    # Step 2: add_pauses arguments
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=0.0,
        help="Minimum gap in seconds for pause detection (default: 0.0).",
    )
    
    # Step 3: classify_gaps arguments
    parser.add_argument(
        "--classify-min-gap",
        type=float,
        default=5.0,
        help="Minimum gap duration in seconds to classify (default: 5.0).",
    )
    
    # General arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for all steps.",
    )
    
    return parser


def run_time_segs(segments_file: Path, args: argparse.Namespace) -> Path:
    """Run 0_time_segs.py to create timeline from segments."""
    print("\n" + "=" * 80)
    print("[STEP 0] Creating timeline from segments")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["time_segs"]),
        "--data",
        str(segments_file),
    ]
    
    if args.timeline_output:
        cmd.extend(["--output", str(args.timeline_output)])
    
    if args.display_timeline:
        cmd.append("--display")
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Determine output file path (using new organized structure)
    if args.timeline_output:
        output_file = Path(args.timeline_output)
    else:
        output_file = segments_file.parent / "3_timeline" / "timeline.json"
    
    if not output_file.exists():
        raise FileNotFoundError(f"Timeline file not created: {output_file}")
    
    print(f"[STEP 0] Timeline created: {output_file}")
    return output_file


def run_close_gaps(timeline_file: Path, args: argparse.Namespace) -> Path:
    """Run 1_close_gaps.py to close small gaps."""
    print("\n" + "=" * 80)
    print("[STEP 1] Closing small gaps in timeline")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["close_gaps"]),
        "--data",
        str(timeline_file),
        "--min-gap",
        str(args.min_gap),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Determine output file path (based on script's output structure)
    output_file = timeline_file.parent / "2_gaps" / "gaps_closed.json"
    
    if not output_file.exists():
        raise FileNotFoundError(f"Gaps closed file not created: {output_file}")
    
    print(f"[STEP 1] Gaps closed: {output_file}")
    return output_file


def run_add_pauses(timeline_file: Path, trajectory_file: Path, args: argparse.Namespace) -> Path:
    """Run 2_add_pauses.py to add pauses."""
    print("\n" + "=" * 80)
    print("[STEP 2] Adding pauses from processed trajectory")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["add_pauses"]),
        "--data",
        str(trajectory_file),
        "--timeline",
        str(timeline_file),
        "--min-gap-seconds",
        str(args.min_gap_seconds),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Determine output file path (based on script's output structure)
    output_file = timeline_file.parent / "3_pauses" / "with_pauses.json"
    
    if not output_file.exists():
        raise FileNotFoundError(f"Pauses file not created: {output_file}")
    
    print(f"[STEP 2] Pauses added: {output_file}")
    return output_file


def run_classify_gaps(timeline_file: Path, args: argparse.Namespace) -> Path:
    """Run 3_classify_gaps.py to classify remaining gaps."""
    print("\n" + "=" * 80)
    print("[STEP 3] Classifying remaining gaps using LLM")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["classify_gaps"]),
        "--data",
        str(timeline_file),
        "--min-gap",
        str(args.classify_min_gap),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Determine output file path (based on script's output structure)
    output_file = timeline_file.parent / "4_classified" / "classified.json"
    
    if not output_file.exists():
        raise FileNotFoundError(f"Classified gaps file not created: {output_file}")
    
    print(f"[STEP 3] Gaps classified: {output_file}")
    return output_file


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        # Step 0: Create timeline from segments
        if not args.skip_time_segs:
            segments_file = args.segments.expanduser().resolve()
            if not segments_file.exists():
                raise FileNotFoundError(f"Segments file not found: {segments_file}")
            
            timeline_file = run_time_segs(segments_file, args)
        else:
            # Use expected output file name or provided timeline output
            if args.timeline_output:
                timeline_file = Path(args.timeline_output).expanduser().resolve()
            else:
                segments_file = args.segments.expanduser().resolve()
                timeline_file = segments_file.parent / "3_timeline" / "timeline.json"
            
            if not timeline_file.exists():
                raise FileNotFoundError(
                    f"Timeline file not found: {timeline_file}. "
                    f"Run timeline creation first or remove --skip-time-segs flag."
                )
            print(f"[SKIP] Using existing timeline file: {timeline_file}")
        
        # Step 1: Close small gaps
        if not args.skip_close_gaps:
            gaps_closed_file = run_close_gaps(timeline_file, args)
        else:
            gaps_closed_file = timeline_file.parent / "2_gaps" / "gaps_closed.json"
            if not gaps_closed_file.exists():
                raise FileNotFoundError(
                    f"Gaps closed file not found: {gaps_closed_file}. "
                    f"Run gap closing first or remove --skip-close-gaps flag."
                )
            print(f"[SKIP] Using existing gaps closed file: {gaps_closed_file}")
        
        # Step 2: Add pauses
        if not args.skip_add_pauses:
            trajectory_file = args.trajectory.expanduser().resolve()
            if not trajectory_file.exists():
                raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
            
            pauses_file = run_add_pauses(gaps_closed_file, trajectory_file, args)
        else:
            pauses_file = gaps_closed_file.parent / "3_pauses" / "with_pauses.json"
            if not pauses_file.exists():
                raise FileNotFoundError(
                    f"Pauses file not found: {pauses_file}. "
                    f"Run pause addition first or remove --skip-add-pauses flag."
                )
            print(f"[SKIP] Using existing pauses file: {pauses_file}")
        
        # Step 3: Classify gaps
        if not args.skip_classify_gaps:
            classified_file = run_classify_gaps(pauses_file, args)
        else:
            print("[SKIP] Skipping gap classification step.")
            classified_file = pauses_file.parent / "4_classified" / "classified.json"
        
        print("\n" + "=" * 80)
        print("[COMPLETE] All timeline processing steps finished successfully!")
        print("=" * 80)
        print(f"\nFinal output: {classified_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Subprocess failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

