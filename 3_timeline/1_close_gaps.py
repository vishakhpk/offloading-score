"""Find and close small gaps between timeline entries."""

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable for utils/language
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    pstr = str(path)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

from utils import (
    calculate_median,
    calculate_percentile,
    format_duration,
    load_json,
    save_json,
)

# Constants
DURATION_RANGES = [
    (0, 5, "< 5s"),
    (5, 10, "5-10s"),
    (10, 30, "10-30s"),
    (30, 60, "30-60s"),
    (60, 120, "1-2min"),
    (120, 300, "2-5min"),
    (300, 600, "5-10min"),
    (600, float("inf"), "> 10min"),
]


def find_gaps(timeline: List[Dict[str, Any]], min_gap_seconds: float = 1.0) -> List[Dict[str, Any]]:
    """Find gaps between timeline entries.
    
    Args:
        timeline: List of timeline entries sorted by start_time
        min_gap_seconds: Minimum gap duration to consider (default: 1 second)
    
    Returns:
        List of gap entries with start, end, duration, and context
    """
    gaps = []
    
    for i in range(len(timeline) - 1):
        current = timeline[i]
        next_entry = timeline[i + 1]
        
        gap_start = current["end_time"]
        gap_end = next_entry["start_time"]
        gap_duration = gap_end - gap_start
        
        if gap_duration >= min_gap_seconds:
            gap = {
                "gap_index": i,
                "start_time": gap_start,
                "end_time": gap_end,
                "duration": gap_duration,
                "before_activity": {
                    "label": current["label"],
                    "annotation": current.get("annotation", ""),
                    "end_time": current["end_time"]
                },
                "after_activity": {
                    "label": next_entry["label"],
                    "annotation": next_entry.get("annotation", ""),
                    "start_time": next_entry["start_time"]
                }
            }
            gaps.append(gap)
    
    return gaps


def close_small_gaps(timeline: List[Dict[str, Any]], threshold: float = 5.0) -> List[Dict[str, Any]]:
    """Close gaps < threshold seconds by extending the previous activity's end_time.
    
    Args:
        timeline: List of timeline entries sorted by start_time
        threshold: Gap duration threshold in seconds (default: 5.0)
    
    Returns:
        Modified timeline with small gaps closed
    """
    if not timeline:
        return timeline
    
    result = []
    
    for i in range(len(timeline)):
        entry = timeline[i].copy()
        
        # Check if there's a gap after this entry
        if i < len(timeline) - 1:
            next_entry = timeline[i + 1]
            gap_duration = next_entry["start_time"] - entry["end_time"]
            
            # If gap is small, extend this entry to close it
            if gap_duration > 0 and gap_duration < threshold:
                entry["end_time"] = next_entry["start_time"]
                entry["duration"] = entry["end_time"] - entry["start_time"]
        
        result.append(entry)
    
    return result


def print_gap_duration_statistics(gaps: List[Dict[str, Any]]) -> None:
    """Print detailed statistics about gap durations.
    
    Args:
        gaps: List of gap dictionaries with duration field
    """
    if not gaps:
        print("No gaps to analyze.")
        return
    
    print("\n--- Gap Duration Statistics ---")
    durations = [g["duration"] for g in gaps]
    durations_sorted = sorted(durations)
    total_gap_time = sum(durations)
    mean_duration = total_gap_time / len(durations) if durations else 0.0
    
    print(f"Total gaps: {len(gaps)}")
    print(f"Total gap time: {format_duration(total_gap_time)}")
    print(f"\nDuration statistics:")
    print(f"  Minimum: {format_duration(min(durations))}")
    print(f"  Maximum: {format_duration(max(durations))}")
    print(f"  Mean: {format_duration(mean_duration)}")
    
    # Median and percentiles
    median = calculate_median(durations_sorted)
    p25 = calculate_percentile(durations_sorted, 0.25)
    p75 = calculate_percentile(durations_sorted, 0.75)
    p90 = calculate_percentile(durations_sorted, 0.90)
    p95 = calculate_percentile(durations_sorted, 0.95)
    
    print(f"  Median: {format_duration(median)}")
    print(f"  25th percentile: {format_duration(p25)}")
    print(f"  75th percentile: {format_duration(p75)}")
    print(f"  90th percentile: {format_duration(p90)}")
    print(f"  95th percentile: {format_duration(p95)}")
    
    # Distribution by duration ranges
    print(f"\nDistribution by duration:")
    for min_dur, max_dur, label in DURATION_RANGES:
        matching_durations = [d for d in durations if min_dur <= d < max_dur]
        if matching_durations:
            count = len(matching_durations)
            total_dur = sum(matching_durations)
            pct = (count / len(durations)) * 100
            print(f"  {label:12s}: {count:4d} gaps ({pct:5.1f}%), {total_dur:8.2f}s ({total_dur/60:6.2f} min)")


def print_remaining_gaps(remaining_gaps: List[Dict[str, Any]], threshold: float = 5.0) -> None:
    """Print details of remaining gaps.
    
    Args:
        remaining_gaps: List of gap dictionaries
        threshold: Minimum gap duration threshold in seconds
    """
    if not remaining_gaps:
        print(f"\nNo remaining gaps (all gaps < {threshold} seconds were closed)")
        return
    
    print(f"\n--- Remaining gaps (>= {threshold} seconds) ---")
    print(f"Total remaining gaps: {len(remaining_gaps)}")
    remaining_durations = [g["duration"] for g in remaining_gaps]
    total_remaining = sum(remaining_durations)
    print(f"Total remaining gap time: {format_duration(total_remaining)}")
    
    print(f"\nRemaining gap details:")
    for i, gap in enumerate(remaining_gaps, 1):
        before_label = gap["before_activity"]["label"]
        after_label = gap["after_activity"]["label"]
        duration = gap["duration"]
        start_time = datetime.fromtimestamp(gap["start_time"]).strftime("%H:%M:%S")
        end_time = datetime.fromtimestamp(gap["end_time"]).strftime("%H:%M:%S")
        
        print(f"  Gap {i}: {format_duration(duration)}")
        print(f"    Between: '{before_label}' -> '{after_label}'")
        print(f"    Time: {start_time} - {end_time}")


def print_time_breakdown(timeline: List[Dict[str, Any]]) -> None:
    """Print time breakdown by activity label.
    
    Args:
        timeline: List of timeline entries with label and duration fields
    """
    print(f"\n--- Time Breakdown After Closing Gaps ---")
    
    label_durations: Dict[str, float] = {}
    label_counts = Counter()
    
    for entry in timeline:
        label = entry.get("label", "unknown")
        duration = entry.get("duration", 0.0)
        label_durations[label] = label_durations.get(label, 0.0) + duration
        label_counts[label] += 1
    
    total_duration = sum(entry.get("duration", 0.0) for entry in timeline)
    print(f"Total timeline duration: {format_duration(total_duration)}")
    print(f"Number of timeline entries: {len(timeline)}")
    
    # Sort by duration (descending)
    sorted_labels = sorted(label_durations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTime breakdown by activity:")
    print(f"{'Activity':<30} {'Segments':<12} {'Duration (s)':<15} {'Duration (min)':<15} {'Percentage':<12}")
    print("-" * 90)
    
    for label, duration in sorted_labels:
        count = label_counts[label]
        percentage = (duration / total_duration) * 100 if total_duration > 0 else 0.0
        print(f"{label:<30} {count:<12} {duration:<15.2f} {duration/60:<15.2f} {percentage:<12.1f}%")
    
    print("-" * 90)
    print(f"{'TOTAL':<30} {len(timeline):<12} {total_duration:<15.2f} {total_duration/60:<15.2f} {100.0:<12.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find gaps between timeline entries and close small gaps."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to timeline JSON file."
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=10.0,
        help="Minimum gap duration in seconds to consider (default: 10.0). Gaps < this threshold will be closed."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress."
    )
    args = parser.parse_args()
    
    # Load timeline
    data_path = Path(args.data)
    timeline = load_json(data_path)
    
    if not timeline:
        raise ValueError(f"No timeline entries found in {data_path}")
    
    # Sort timeline by start_time if not already sorted
    timeline = sorted(timeline, key=lambda x: x["start_time"])
    
    if args.verbose:
        print(f"Processing {len(timeline)} timeline entries...")
    
    # Find gaps
    gaps = find_gaps(timeline, min_gap_seconds=args.min_gap)

    # Organize outputs into subdirectories
    # Find the issue directory (walk up past processing step directories)
    current = data_path.parent
    while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified"]:
        current = current.parent
    output_base = current / "3_timeline" / "2_gaps"
    output_base.mkdir(parents=True, exist_ok=True)

    if not gaps:
        print("No gaps found.")
        timeline_path = output_base / "gaps_closed.json"
        save_json(timeline, timeline_path)
        print(f"Saved timeline with closed gaps to: {timeline_path}")
        print_time_breakdown(timeline)
        return
    
    if args.verbose:
        total_gap_time = sum(g["duration"] for g in gaps)
        print(f"Found {len(gaps)} gaps totaling {total_gap_time:.2f} seconds ({total_gap_time/60:.2f} minutes)")
    
    # Print gap duration statistics
    print_gap_duration_statistics(gaps)
    
    # Automatically close gaps < threshold
    print(f"\n--- Closing gaps < {args.min_gap} seconds ---")
    timeline_closed = close_small_gaps(timeline, threshold=args.min_gap)
    
    # Count how many gaps were closed
    closed_gaps = [g for g in gaps if g["duration"] < args.min_gap]
    closed_count = len(closed_gaps)
    closed_duration = sum(g["duration"] for g in closed_gaps)
    print(f"Closed {closed_count} gaps totaling {format_duration(closed_duration)}")
    
    # Find remaining gaps (>= threshold)
    remaining_gaps = find_gaps(timeline_closed, min_gap_seconds=args.min_gap)
    print_remaining_gaps(remaining_gaps, threshold=args.min_gap)
    
    # Save timeline with closed gaps
    timeline_path = output_base / "gaps_closed.json"
    save_json(timeline_closed, timeline_path)
    print(f"\nSaved timeline with closed gaps to: {timeline_path}")
    
    # Save remaining gaps if any
    if remaining_gaps:
        output_path = output_base / "remaining_gaps.json"
        save_json(remaining_gaps, output_path)
        print(f"Saved remaining gaps to: {output_path}")
    
    # Print time breakdown after closing gaps
    print_time_breakdown(timeline_closed)


if __name__ == "__main__":
    main()
