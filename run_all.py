"""
Run all processing steps in sequence.

This script orchestrates the entire pipeline:
0. Preprocessing:
   0.0. Data trimming (optional): Trim overlapping issue folders
   0.1. Get trajectory: Extract trajectory from screenshots and states
   0.2. Get AI interactions (optional): Extract AI interactions from logs
   0.3. Summarize issue (optional): Fetch and summarize issue from GitHub
1. Segmentation:
   1.1. Segment: Split trajectory into segments
   1.2. Annotate: Annotate segments with activity labels
2. Induction:
   2.1. Induce: Induce high-level labels from segments
   2.2. Postprocess: Merge adjacent segments with same labels
3. Timeline:
   3.1. Create timeline: Map segments to timeline entries
   3.2. Close gaps: Close small gaps between activities
   3.3. Add pauses: Add detected pauses from trajectory
   3.4. Classify gaps: Classify remaining gaps using LLM
4. Visualization:
   4.1. Generate visualizations: Create timeline and breakdown charts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Script paths
PREPROCESSING_SCRIPTS = {
    "data_trimming": ROOT / "0_preprocessing" / "0_data_trimming.py",
    "get_trajectory": ROOT / "0_preprocessing" / "1_get_trajectory.py",
    "get_ai_interactions": ROOT / "0_preprocessing" / "2_get_ai_interactions.py",
    "summarize_issue": ROOT / "0_preprocessing" / "3_summarize_issue.py",
}

SEGMENT_SCRIPTS = {
    "segment": ROOT / "1_segment" / "0_segment.py",
    "annotate": ROOT / "1_segment" / "1_annotate.py",
}

INDUCTION_SCRIPTS = {
    "induce": ROOT / "2_induction" / "0_induce.py",
    "postprocess": ROOT / "2_induction" / "1_postprocessing.py",
}

TIMELINE_SCRIPTS = {
    "time_segs": ROOT / "3_timeline" / "0_time_segs.py",
    "close_gaps": ROOT / "3_timeline" / "1_close_gaps.py",
    "add_pauses": ROOT / "3_timeline" / "2_add_pauses.py",
    "classify_gaps": ROOT / "3_timeline" / "3_classify_gaps.py",
}

VISUALIZE_SCRIPT = ROOT / "4_visualize" / "visualize.py"


def detect_completed_steps(data_dir: Path) -> dict[str, bool]:
    """Detect which processing steps have already been completed.
    
    Returns:
        Dictionary with keys indicating which steps are complete:
        - preprocessing: True if processed_trajectory.json exists
        - segmentation: True if annotated.json exists
        - induction: True if merged.json exists
        - timeline: True if classified.json exists
        - visualization: True if visualizations directory exists
    """
    data_dir = data_dir.expanduser().resolve()
    
    # Determine base directory (where outputs are stored)
    # Try common locations for trajectory to determine base dir
    base_dir = data_dir.parent
    trajectory_path = data_dir.parent / "0_preprocessing" / "processed_trajectory.json"
    
    completed = {
        "preprocessing": False,
        "segmentation": False,
        "induction": False,
        "timeline": False,
        "visualization": False,
    }
    
    # Check preprocessing
    if trajectory_path.exists():
        completed["preprocessing"] = True
    
    # Check segmentation (both segments and annotated files should exist)
    segments_path = base_dir / "1_segment" / "segments.json"
    annotated_path = base_dir / "1_segment" / "annotated.json"
    
    if annotated_path.exists():
        completed["segmentation"] = True
    elif segments_path.exists():
        # Segments exist but not annotated - segmentation is partially done
        # Don't mark as complete, so annotation will run
        pass
    
    # Check induction
    merged_path = base_dir / "2_induction" / "merged.json"
    print(merged_path)
    if merged_path.exists():
        completed["induction"] = True
    
    # Check timeline
    classified_path = base_dir / "3_timeline" / "4_classified" / "classified.json"
    if classified_path.exists():
        completed["timeline"] = True
    
    # Check visualization (look for visualizations directory in multiple possible locations)
    vis_dir1 = base_dir / "5_visualizations"
    vis_dir2 = base_dir / "3_timeline" / "5_visualizations"
    if (vis_dir1.exists() and any(vis_dir1.iterdir())) or (vis_dir2.exists() and any(vis_dir2.iterdir())):
        completed["visualization"] = True
    
    return completed


def apply_resume_flags(args: argparse.Namespace, data_dir: Path) -> None:
    """Apply resume logic: automatically set skip flags based on completed steps."""
    completed = detect_completed_steps(data_dir)
    
    print("\n" + "=" * 80)
    print("[RESUME] Detecting completed steps...")
    print("=" * 80)
    
    for step, is_complete in completed.items():
        status = "✓ COMPLETE" if is_complete else "✗ PENDING"
        print(f"  {step:15s}: {status}")
    
    # Set skip flags based on what's completed
    if completed["preprocessing"]:
        args.skip_preprocessing = True
        print("\n[RESUME] Skipping preprocessing (already completed)")
    
    if completed["segmentation"]:
        args.skip_segmentation = True
        print("[RESUME] Skipping segmentation (already completed)")
    
    if completed["induction"]:
        args.skip_induction = True
        print("[RESUME] Skipping induction (already completed)")
    
    if completed["timeline"]:
        args.skip_timeline = True
        print("[RESUME] Skipping timeline (already completed)")
    
    if completed["visualization"]:
        args.skip_visualization = True
        print("[RESUME] Skipping visualization (already completed)")
    
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all processing steps in sequence."
    )
    
    # Required arguments
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data directory containing screenshots, states, etc."
    )
    
    # Optional arguments for skipping steps
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing steps (use existing processed_trajectory.json)."
    )
    parser.add_argument(
        "--skip-data-trimming",
        action="store_true",
        help="Skip data trimming step (only needed when processing multiple issues)."
    )
    parser.add_argument(
        "--skip-summarize-issue",
        action="store_true",
        help="Skip issue summarization step."
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip segmentation and annotation steps."
    )
    parser.add_argument(
        "--skip-induction",
        action="store_true",
        help="Skip label induction and postprocessing steps."
    )
    parser.add_argument(
        "--skip-timeline",
        action="store_true",
        help="Skip timeline processing steps."
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation."
    )
    
    # Preprocessing arguments
    parser.add_argument(
        "--issues-root",
        type=Path,
        default=None,
        help="Parent directory containing multiple issue_* folders (for data trimming and issue summarization). If not provided, these steps are skipped."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing log files for AI interactions (default: auto-detect .specstory, else <data-dir>/logs)."
    )
    parser.add_argument(
        "--summarize-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for issue summarization (default: gpt-4o)."
    )
    
    # Segmentation arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="MSE threshold for splitting segments."
    )
    parser.add_argument(
        "--auto-percentile",
        type=float,
        default=75.0,
        help="Derive threshold from this percentile of scores (default: 75.0)."
    )
    parser.add_argument(
        "--no-remerge",
        dest="do_remerge",
        action="store_false",
        help="Disable LLM-based re-merging of adjacent segments (default: enabled)."
    )
    parser.set_defaults(do_remerge=True)
    
    # Timeline arguments
    parser.add_argument(
        "--min-gap",
        type=float,
        default=5.0,
        help="Minimum gap duration in seconds (default: 5.0)."
    )
    
    # General arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for all steps."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically skip steps that have already been completed (resume from last checkpoint)."
    )
    
    return parser


def run_preprocessing(data_dir: Path, args: argparse.Namespace) -> Path:
    """Run preprocessing steps: trim data, get trajectory, and AI interactions."""
    print("\n" + "=" * 80)
    print("[STEP 0] PREPROCESSING")
    print("=" * 80)
    
    data_dir = data_dir.expanduser().resolve()
    # Prefer explicit log_dir; else auto-detect .specstory (issue root or data_dir); fallback to logs
    if args.log_dir:
        log_dir = args.log_dir.expanduser().resolve()
    else:
        specstory_candidates = [
            data_dir.parent / ".specstory",
            data_dir / ".specstory",
        ]
        log_dir = next((p for p in specstory_candidates if p.exists()), data_dir / "logs")
    
    # Step 0.0: Data trimming (runs automatically if issues_root provided, otherwise tries parent of data_dir)
    if not args.skip_data_trimming:
        print("\n[0.0] Trimming overlapping issue data...")
        if args.issues_root:
            issues_root = args.issues_root.expanduser().resolve()
        else:
            # Try to use parent of data_dir as issues root
            issues_root = data_dir.parent
            print(f"  Using parent directory as issues root: {issues_root}")
        
        cmd = [
            sys.executable,
            str(PREPROCESSING_SCRIPTS["data_trimming"]),
            "--data",
            str(issues_root),
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[0.0] Data trimming complete")
    
    # Step 0.1: Get trajectory
    # Note: 1_get_trajectory.py only accepts --data and --output_path
    # It automatically looks for screenshots and states in subdirectories
    print("\n[0.1] Getting trajectory from screenshots and states...")
    cmd = [
        sys.executable,
        str(PREPROCESSING_SCRIPTS["get_trajectory"]),
        "--data",
        str(data_dir),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # 1_get_trajectory.py saves to os.path.dirname(args.data) / "0_preprocessing"
    # So if data_dir is the "data" subdirectory, it saves to the parent (issue directory)
    # Check the parent directory first (most common case)
    trajectory_path = data_dir.parent / "0_preprocessing" / "processed_trajectory.json"
    if not trajectory_path.exists():
        # Try the alternative location (if data_dir is already the issue directory)
        trajectory_path = data_dir / "0_preprocessing" / "processed_trajectory.json"
        if not trajectory_path.exists():
            raise FileNotFoundError(
                f"Trajectory file not found. Checked:\n"
                f"  - {data_dir.parent / '0_preprocessing' / 'processed_trajectory.json'}\n"
                f"  - {data_dir / '0_preprocessing' / 'processed_trajectory.json'}"
            )
    
    print(f"[0.1] Trajectory created: {trajectory_path}")
    
    # Step 0.2: Get AI interactions (optional, if log_dir exists)
    if log_dir.exists():
        print("\n[0.2] Extracting AI interactions from logs...")
        cmd = [
            sys.executable,
            str(PREPROCESSING_SCRIPTS["get_ai_interactions"]),
            "--data",
            str(log_dir),
        ]
        
        if args.verbose:
            cmd.append("--verbose")
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[0.2] AI interactions extracted")
    else:
        print(f"[0.2] Skipping AI interactions (log directory not found: {log_dir})")
    
    # Step 0.3: Summarize issue (optional, if issues_root is provided)
    if not args.skip_summarize_issue and args.issues_root:
        print("\n[0.3] Summarizing issue from GitHub...")
        issues_root = args.issues_root.expanduser().resolve()
        cmd = [
            sys.executable,
            str(PREPROCESSING_SCRIPTS["summarize_issue"]),
            "--data",
            str(issues_root),
            "--model",
            args.summarize_model,
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[0.3] Issue summarization complete")
    elif not args.skip_summarize_issue:
        print("\n[0.3] Skipping issue summarization (--issues-root not provided)")
    
    return trajectory_path


def run_segmentation(trajectory_path: Path, args: argparse.Namespace) -> Path:
    """Run segmentation and annotation steps."""
    print("\n" + "=" * 80)
    print("[STEP 1] SEGMENTATION AND ANNOTATION")
    print("=" * 80)
    
    # Step 1.1: Segment
    print("\n[1.1] Segmenting trajectory...")
    cmd = [
        sys.executable,
        str(SEGMENT_SCRIPTS["segment"]),
        "--trajectory",
        str(trajectory_path),
    ]
    
    if args.threshold is not None:
        cmd.extend(["--threshold", str(args.threshold)])
    else:
        cmd.extend(["--auto-percentile", str(args.auto_percentile)])
    
    if args.do_remerge:
        cmd.append("--do-remerge")
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    segments_path = trajectory_path.parent.parent / "1_segment" / "segments.json"
    if not segments_path.exists():
        raise FileNotFoundError(f"Segments file not created: {segments_path}")
    
    print(f"[1.1] Segments created: {segments_path}")
    
    # Step 1.2: Annotate
    print("\n[1.2] Annotating segments...")
    cmd = [
        sys.executable,
        str(SEGMENT_SCRIPTS["annotate"]),
        "--input",
        str(segments_path),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    annotated_path = trajectory_path.parent.parent / "1_segment" / "annotated.json"
    if not annotated_path.exists():
        raise FileNotFoundError(f"Annotated file not created: {annotated_path}")
    
    print(f"[1.2] Segments annotated: {annotated_path}")
    
    return annotated_path


def run_induction(annotated_path: Path, args: argparse.Namespace) -> Path:
    """Run label induction and postprocessing steps."""
    print("\n" + "=" * 80)
    print("[STEP 2] LABEL INDUCTION AND POSTPROCESSING")
    print("=" * 80)
    
    # Step 2.1: Induce labels
    print("\n[2.1] Inducing labels from segments...")
    cmd = [
        sys.executable,
        str(INDUCTION_SCRIPTS["induce"]),
        "--data",
        str(annotated_path),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    grouped_path = annotated_path.parent / "2_induction" / "grouped.json"
    if not grouped_path.exists():
        raise FileNotFoundError(f"Grouped file not created: {grouped_path}")
    
    print(f"[2.1] Labels induced: {grouped_path}")
    
    # Step 2.2: Postprocess (merge adjacent same labels)
    print("\n[2.2] Postprocessing (merging adjacent same labels)...")
    cmd = [
        sys.executable,
        str(INDUCTION_SCRIPTS["postprocess"]),
        "--data",
        str(grouped_path),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    merged_path = annotated_path.parent.parent / "2_induction" / "merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged file not created: {merged_path}")
    
    print(f"[2.2] Segments merged: {merged_path}")
    
    return merged_path


def run_timeline(merged_path: Path, trajectory_path: Path, args: argparse.Namespace) -> Path:
    """Run timeline processing steps."""
    print("\n" + "=" * 80)
    print("[STEP 3] TIMELINE PROCESSING")
    print("=" * 80)
    
    # Step 3.1: Create timeline
    print("\n[3.1] Creating timeline from segments...")
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["time_segs"]),
        "--data",
        str(merged_path),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Find the issue directory (walk up past processing step directories)
    current = merged_path.parent
    while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified"]:
        current = current.parent
    timeline_path = current / "3_timeline" / "timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline file not created: {timeline_path}")
    
    print(f"[3.1] Timeline created: {timeline_path}")
    
    # Step 3.2: Close small gaps
    print("\n[3.2] Closing small gaps...")
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["close_gaps"]),
        "--data",
        str(timeline_path),
        "--min-gap",
        str(args.min_gap),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    gaps_closed_path = timeline_path.parent / "2_gaps" / "gaps_closed.json"
    if not gaps_closed_path.exists():
        raise FileNotFoundError(f"Gaps closed file not created: {gaps_closed_path}")
    
    print(f"[3.2] Gaps closed: {gaps_closed_path}")
    
    # Step 3.3: Add pauses
    print("\n[3.3] Adding pauses from trajectory...")
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["add_pauses"]),
        "--data",
        str(trajectory_path),
        "--timeline",
        str(gaps_closed_path),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    pauses_path = timeline_path.parent / "3_pauses" / "with_pauses.json"
    if not pauses_path.exists():
        raise FileNotFoundError(f"Pauses file not created: {pauses_path}")
    
    print(f"[3.3] Pauses added: {pauses_path}")
    
    # Step 3.4: Classify gaps
    print("\n[3.4] Classifying remaining gaps...")
    cmd = [
        sys.executable,
        str(TIMELINE_SCRIPTS["classify_gaps"]),
        "--data",
        str(pauses_path),
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    classified_path = timeline_path.parent / "4_classified" / "classified.json"
    if not classified_path.exists():
        raise FileNotFoundError(f"Classified file not created: {classified_path}")
    
    print(f"[3.4] Gaps classified: {classified_path}")
    
    return classified_path


def run_visualization(timeline_path: Path, args: argparse.Namespace) -> None:
    """Run visualization generation."""
    print("\n" + "=" * 80)
    print("[STEP 4] VISUALIZATION")
    print("=" * 80)
    
    print("\n[4.1] Generating timeline visualizations...")
    cmd = [
        sys.executable,
        str(VISUALIZE_SCRIPT),
        "--data",
        str(timeline_path),
        "--compress-pauses"
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"[4.1] Visualizations generated")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        data_dir = args.data.expanduser().resolve()
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Auto-detect issues_root if not provided: use parent of data_dir
        if args.issues_root is None:
            args.issues_root = data_dir.parent.parent
            if args.verbose:
                print(f"[AUTO] Using parent as issues_root: {args.issues_root}")
        
        # Apply resume logic if requested
        if args.resume:
            apply_resume_flags(args, data_dir)
        # Step 0: Preprocessing
        trajectory_path = data_dir.parent / "0_preprocessing" / "processed_trajectory.json"
        if args.skip_preprocessing and trajectory_path.exists():
            print(f"[SKIP] Using existing trajectory: {trajectory_path}")
        else:
            trajectory_path = run_preprocessing(data_dir, args)
        
        # Step 1: Segmentation
        if not args.skip_segmentation:
            annotated_path = run_segmentation(trajectory_path, args)
        else:
            annotated_path = data_dir.parent / "1_segment" / "annotated.json"
            segments_path = data_dir.parent / "1_segment" / "segments.json"
            
            if not annotated_path.exists():
                # Check if segments exist but annotation is missing - run annotation only
                if segments_path.exists():
                    print(f"[SKIP] Segmentation skipped, but found segments. Running annotation only...")
                    print("\n[1.2] Annotating segments...")
                    cmd = [
                        sys.executable,
                        str(SEGMENT_SCRIPTS["annotate"]),
                        "--input",
                        str(segments_path),
                    ]
                    
                    if args.verbose:
                        cmd.append("--verbose")
                    
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    
                    if not annotated_path.exists():
                        raise FileNotFoundError(f"Annotated file not created: {annotated_path}")
                    
                    print(f"[1.2] Segments annotated: {annotated_path}")
                else:
                    raise FileNotFoundError(
                        f"Annotated file not found: {annotated_path}. "
                        f"Segments file also not found: {segments_path}. "
                        f"Run segmentation first or remove --skip-segmentation flag."
                    )
            else:
                print(f"[SKIP] Using existing annotated segments: {annotated_path}")
        
        # Step 2: Induction
        merged_path = data_dir.parent / "2_induction" / "merged.json"
        if args.skip_induction and merged_path.exists():
            print(f"[SKIP] Using existing merged segments: {merged_path}")
        else:
            merged_path = run_induction(annotated_path, args)
        
        # Step 3: Timeline
        classified_path = data_dir.parent / "3_timeline" / "4_classified" / "classified.json"
        if args.skip_timeline and classified_path.exists():
            print(f"[SKIP] Using existing classified timeline: {classified_path}")
        else:
            classified_path = run_timeline(merged_path, trajectory_path, args)
        
        # Step 4: Visualization
        if not args.skip_visualization:
            run_visualization(classified_path, args)
        else:
            print("[SKIP] Skipping visualization generation.")
        
        print("\n" + "=" * 80)
        print("[COMPLETE] All processing steps finished successfully!")
        print("=" * 80)
        print(f"\nFinal output: {classified_path}")
        print(f"Visualizations: {classified_path.parent.parent / '5_visualizations'}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Subprocess failed with return code {e.returncode}")
        print(f"\n[TIP] To resume from where it stopped, run with --resume flag:")
        print(f"     python run_all.py --data {args.data} --resume")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print(f"\n[TIP] To resume from where it stopped, run with --resume flag:")
        print(f"     python run_all.py --data {args.data} --resume")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[TIP] To resume from where it stopped, run with --resume flag:")
        print(f"     python run_all.py --data {args.data} --resume")
        sys.exit(1)



if __name__ == "__main__":
    main()
