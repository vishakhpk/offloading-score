import argparse
import sys
from pathlib import Path
from typing import Any
from datetime import datetime

# Ensure project root is importable for utils/language
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    pstr = str(path)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

from utils import extract_segments, load_json, save_json


def get_segment_time(segment: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract start and end time from a segment.
    
    Returns:
        (start_time, end_time) in Unix timestamp, or (None, None) if not available
    """

    from language import get_first_action, get_last_action
    
    first_action = get_first_action(segment)
    last_action = get_last_action(segment)
    
    start = None
    end = None
    
    if first_action and first_action.get("time"):
        start = first_action["time"].get("before")
    if last_action and last_action.get("time"):
        end = last_action["time"].get("after")
    
    return (start, end)


def create_timeline(segments: list[dict[str, Any]], verbose: bool = False) -> list[dict[str, Any]]:
    """Create a timeline mapping from labeled segments.
    
    Returns:
        List of timeline entries with start, end, label, and duration
    """
    timeline = []
    
    for i, segment in enumerate(segments):
        start_time, end_time = get_segment_time(segment)
        
        if start_time is None or end_time is None:
            if verbose:
                print(f"Warning: Segment {i} has no time information, skipping")
            continue
        
        label = segment.get("label", "unknown")
        annotation = segment.get("annotation", "")
        
        duration = end_time - start_time
        
        timeline_entry = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "label": label,
            "annotation": annotation
        }
        
        timeline.append(timeline_entry)
    
    return timeline


def format_timeline_for_display(timeline: list[dict[str, Any]]) -> str:
    """Format timeline for human-readable display."""
    lines = []
    lines.append("=" * 80)
    lines.append("TIMELINE OF ACTIVITIES")
    lines.append("=" * 80)
    
    if not timeline:
        return "No timeline data available."
    
    # Overall statistics
    total_duration = timeline[-1]["end_time"] - timeline[0]["start_time"]
    start_dt = datetime.fromtimestamp(timeline[0]["start_time"])
    end_dt = datetime.fromtimestamp(timeline[-1]["end_time"])
    
    lines.append(f"\nTotal Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    lines.append(f"Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"End: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Segments: {len(timeline)}")
    
    # Timeline entries summary
    lines.append("\n" + "-" * 80)
    lines.append("TIMELINE ENTRIES")
    lines.append("-" * 80)
    
    for entry in timeline:
        start_dt = datetime.fromtimestamp(entry["start_time"])
        end_dt = datetime.fromtimestamp(entry["end_time"])
        duration = entry["duration"]
        
        lines.append(f"\n{entry['label']}")
        lines.append(f"  Time: {start_dt.strftime('%H:%M:%S')} - {end_dt.strftime('%H:%M:%S')}")
        lines.append(f"  Duration: {duration:.2f}s ({duration/60:.2f} min)")
        
        # Show annotation
        annotation = entry.get("annotation", "")
        if annotation:
            lines.append(f"  Annotation: {annotation}")
    
    return "\n".join(lines)


def print_timeline_statistics(timeline: list[dict[str, Any]], title: str = "Timeline Statistics") -> None:
    """Print activity breakdown statistics for a timeline.
    
    Args:
        timeline: List of timeline entries with start_time, end_time, duration, and label
        title: Title for the statistics section
    """
    if not timeline:
        print(f"\n--- {title} ---")
        print("No timeline data available.")
        return
    
    from collections import Counter
    
    print(f"\n--- {title} ---")
    label_counts = Counter(entry["label"] for entry in timeline)
    
    total_accounted_duration = 0.0
    for label, count in label_counts.most_common():
        total_duration = sum(e["duration"] for e in timeline if e["label"] == label)
        total_accounted_duration += total_duration
        print(f"  {label}: {count} segments, {total_duration:.2f}s ({total_duration/60:.2f} min)")
    
    # Total timeline span
    sorted_timeline = sorted(timeline, key=lambda x: x["start_time"])
    total_timeline_span = sorted_timeline[-1]["end_time"] - sorted_timeline[0]["start_time"]
    print(f"\nTotal Timeline Span: {total_timeline_span:.2f}s ({total_timeline_span/60:.2f} min)")
    
    # Unaccounted time (gaps between segments)
    unaccounted_time = total_timeline_span - total_accounted_duration
    if unaccounted_time > 0:
        print(f"Unaccounted Time (gaps): {unaccounted_time:.2f}s ({unaccounted_time/60:.2f} min)")
        if total_timeline_span > 0:
            percentage = (unaccounted_time / total_timeline_span) * 100
            print(f"  ({percentage:.1f}% of total timeline span)")
    else:
        print(f"Unaccounted Time: 0.0s (all time is accounted for)")


def main():
    parser = argparse.ArgumentParser(
        description="Map labeled segments to a timeline."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to labeled segments JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for timeline JSON (default: input path with _timeline suffix)."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Print human-readable timeline to stdout."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress."
    )
    args = parser.parse_args()
    
    # Load labeled segments
    data_path = Path(args.data)
    data = load_json(data_path)
    segments = extract_segments(data)
    
    if not segments:
        raise ValueError(f"No segments found in {data_path}")
    
    if args.verbose:
        print(f"Processing {len(segments)} segments...")
    
    # Create timeline
    timeline = create_timeline(segments, verbose=args.verbose)
    
    if not timeline:
        raise ValueError("No segments with time information found.")
    
    if args.verbose:
        print(f"Created timeline with {len(timeline)} entries")
    
    # Organize outputs into subdirectories
    # Find the issue directory (walk up past processing step directories)
    current = data_path.parent
    while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified"]:
        current = current.parent
    output_base = current / "3_timeline"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save timeline
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_base / "timeline.json"
    
    save_json(timeline, output_path)
    print(f"\nSaved timeline to: {output_path}")
    
    # Display timeline if requested
    if args.display:
        print("\n" + format_timeline_for_display(timeline))
    
    # Print summary statistics
    print_timeline_statistics(timeline)
    

if __name__ == "__main__":
    main()

