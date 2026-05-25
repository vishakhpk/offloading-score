"""Report pauses (ctrl+c followed by large gaps) in a processed trajectory JSON file."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable for utils/language
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    pstr = str(path)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

from utils import load_json, save_json

# Import print_timeline_statistics from 0_time_segs.py
import importlib.util
_time_segs_spec = importlib.util.spec_from_file_location("time_segs", Path(__file__).parent / "0_time_segs.py")
_time_segs = importlib.util.module_from_spec(_time_segs_spec)
_time_segs_spec.loader.exec_module(_time_segs)
print_timeline_statistics = _time_segs.print_timeline_statistics


def load_nodes(path: str) -> List[Dict[str, Any]]:
    """Load all nodes from the processed trajectory file."""
    data = load_json(Path(path))
    nodes = data.get("nodes", [])
    
    # Ensure chronological order by `time.before` with fallback to index order.
    nodes.sort(key=lambda node: (node.get("time", {}).get("before", float("inf"))))
    return nodes


def extract_time(node: Dict[str, Any], key: str) -> Optional[float]:
    time_info = node.get("time", {})
    value = time_info.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def compute_pauses(nodes: List[Dict[str, Any]], min_gap_minutes: float = 2.0) -> List[Dict[str, Any]]:
    """Compute pauses where the last action is ctrl+c and there's a large gap > min_gap_minutes."""
    results: List[Dict[str, Any]] = []
    min_gap_seconds = min_gap_minutes * 60.0

    for idx in range(len(nodes) - 1):
        prev = nodes[idx]
        nxt = nodes[idx + 1]

        # Check if previous action contains "ctrl+c" (case-insensitive)
        prev_action_str = str(prev.get("action", "")).lower()
        if "ctrl+c" not in prev_action_str:
            continue

        prev_after = extract_time(prev, "after")
        next_before = extract_time(nxt, "before")

        if prev_after is None or next_before is None:
            continue

        gap_seconds = next_before - prev_after
        if gap_seconds <= min_gap_seconds:
            continue

        prev_action = prev.get("action", "")
        next_action = nxt.get("action", "")
        
        # Format similar to timeline JSON
        results.append(
            {
                "start_time": prev_after,
                "end_time": next_before,
                "duration": gap_seconds,
                "label": "pause",
                "annotation": f"Pause after ctrl+c action: {prev_action}. Next action: {next_action}",
            }
        )

    # Sort by start_time (chronological order)
    results.sort(key=lambda entry: entry["start_time"])
    return results


def compute_large_gaps(nodes: List[Dict[str, Any]], min_gap_minutes: float = 3.0) -> List[Dict[str, Any]]:
    """Find all gaps > min_gap_minutes between consecutive actions in the trajectory.
    
    Args:
        nodes: List of trajectory nodes
        min_gap_minutes: Minimum gap duration in minutes (default: 3.0)
    
    Returns:
        List of gap entries with start_time, end_time, duration, label, and annotation
    """
    results: List[Dict[str, Any]] = []
    min_gap_seconds = min_gap_minutes * 60.0

    for idx in range(len(nodes) - 1):
        prev = nodes[idx]
        nxt = nodes[idx + 1]

        prev_after = extract_time(prev, "after")
        next_before = extract_time(nxt, "before")

        if prev_after is None or next_before is None:
            continue

        gap_seconds = next_before - prev_after
        if gap_seconds <= min_gap_seconds:
            continue

        prev_action = prev.get("action", "")
        next_action = nxt.get("action", "")
        
        # Format similar to timeline JSON
        results.append(
            {
                "start_time": prev_after,
                "end_time": next_before,
                "duration": gap_seconds,
                "label": "gap",
                "annotation": f"Gap between actions: '{prev_action}' -> '{next_action}'",
            }
        )

    # Sort by start_time (chronological order)
    results.sort(key=lambda entry: entry["start_time"])
    return results


def calculate_overlap(seg1: Dict[str, Any], seg2: Dict[str, Any]) -> float:
    """Calculate overlap duration between two segments in seconds.
    
    Args:
        seg1: First segment with start_time and end_time
        seg2: Second segment with start_time and end_time
        
    Returns:
        Overlap duration in seconds, or 0.0 if no overlap
    """
    start1 = seg1["start_time"]
    end1 = seg1["end_time"]
    start2 = seg2["start_time"]
    end2 = seg2["end_time"]
    
    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    if overlap_start < overlap_end:
        return overlap_end - overlap_start
    return 0.0


def check_overlaps(timeline: List[Dict[str, Any]], pauses: List[Dict[str, Any]]) -> None:
    """Check for overlaps between pauses and timeline segments and print them."""
    overlaps_found = []
    
    for pause in pauses:
        for segment in timeline:
            overlap = calculate_overlap(pause, segment)
            
            if overlap > 0:
                overlaps_found.append({
                    "pause": pause,
                    "segment": segment,
                    "overlap_seconds": overlap,
                    "overlap_minutes": overlap / 60.0
                })
    
    if overlaps_found:
        print(f"\nFound {len(overlaps_found)} overlap(s) between pauses and timeline segments:")
        print("=" * 80)
        
        for idx, overlap_info in enumerate(overlaps_found, 1):
            pause = overlap_info["pause"]
            segment = overlap_info["segment"]
            overlap_sec = overlap_info["overlap_seconds"]
            overlap_min = overlap_info["overlap_minutes"]
            
            print(f"\nOverlap {idx}:")
            print(f"  Pause: {pause['start_time']:.2f} - {pause['end_time']:.2f} "
                  f"({pause['duration']/60:.2f} min)")
            print(f"  Segment: {segment['start_time']:.2f} - {segment['end_time']:.2f} "
                  f"({segment['duration']/60:.2f} min, label: {segment['label']})")
            print(f"  Overlap: {overlap_sec:.2f} seconds ({overlap_min:.2f} minutes)")
            print(f"  Pause annotation: {pause['annotation']}")
    else:
        print("\nNo overlaps found between pauses and timeline segments.")


def split_segment_by_gap(segment: Dict[str, Any], gap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split a timeline segment if it overlaps with a gap.
    
    Args:
        segment: Timeline segment with start_time and end_time
        gap: Gap entry with start_time and end_time
    
    Returns:
        List of segments (may be 0, 1, or 2 segments depending on overlap)
    """
    seg_start = segment["start_time"]
    seg_end = segment["end_time"]
    gap_start = gap["start_time"]
    gap_end = gap["end_time"]
    
    # No overlap
    if seg_end <= gap_start or seg_start >= gap_end:
        return [segment]
    
    # Gap completely covers segment
    if gap_start <= seg_start and gap_end >= seg_end:
        return []  # Segment is completely overlapped, remove it
    
    # Gap is completely inside segment - split into two parts
    if gap_start > seg_start and gap_end < seg_end:
        before = segment.copy()
        before["end_time"] = gap_start
        before["duration"] = gap_start - seg_start
        
        after = segment.copy()
        after["start_time"] = gap_end
        after["duration"] = seg_end - gap_end
        
        return [before, after]
    
    # Gap overlaps start of segment
    if gap_start <= seg_start and gap_end < seg_end:
        after = segment.copy()
        after["start_time"] = gap_end
        after["duration"] = seg_end - gap_end
        return [after]
    
    # Gap overlaps end of segment
    if gap_start > seg_start and gap_end >= seg_end:
        before = segment.copy()
        before["end_time"] = gap_start
        before["duration"] = gap_start - seg_start
        return [before]
    
    return [segment]  # Should not reach here


def split_timeline_by_gaps(
    timeline: List[Dict[str, Any]], 
    gaps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Split timeline segments that overlap with gaps.
    
    Args:
        timeline: List of timeline segments
        gaps: List of gap entries
    
    Returns:
        Modified timeline with segments split around gaps
    """
    if not gaps:
        return timeline
    
    result = []
    
    for segment in timeline:
        segments_to_add = [segment]
        
        # Check against all gaps
        for gap in gaps:
            new_segments = []
            for seg in segments_to_add:
                split_segs = split_segment_by_gap(seg, gap)
                new_segments.extend(split_segs)
            segments_to_add = new_segments
        
        result.extend(segments_to_add)
    
    return result


def merge_timeline_and_pauses(
    timeline: List[Dict[str, Any]], 
    pauses: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge pauses into timeline, sorted by start_time."""
    merged = timeline + pauses
    merged.sort(key=lambda x: x["start_time"])
    return merged


def deduplicate_gaps(gaps1: List[Dict[str, Any]], gaps2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove gaps from gaps2 that overlap with gaps in gaps1.
    
    Args:
        gaps1: First list of gaps (e.g., pauses)
        gaps2: Second list of gaps (e.g., large gaps)
    
    Returns:
        Filtered gaps2 with overlapping gaps removed
    """
    if not gaps1 or not gaps2:
        return gaps2
    
    # Create a set of (start_time, end_time) tuples from gaps1 for quick lookup
    gaps1_set = {(g["start_time"], g["end_time"]) for g in gaps1}
    
    # Filter gaps2 to remove those that match gaps1
    filtered = []
    for gap in gaps2:
        gap_key = (gap["start_time"], gap["end_time"])
        if gap_key not in gaps1_set:
            filtered.append(gap)
    
    return filtered




def save_gap_report(gaps: List[Dict[str, Any]], output_path: str) -> None:
    save_json(gaps, Path(output_path))
    print(f"Saved {len(gaps)} gaps to {output_path}")


def print_pause_summary(pauses: List[Dict[str, Any]]) -> None:
    if not pauses:
        print("No pauses detected (ctrl+c followed by gap > threshold).")
        return

    # Sort by duration (descending) for display
    sorted_pauses = sorted(pauses, key=lambda p: p["duration"], reverse=True)

    for idx, pause in enumerate(sorted_pauses):
        duration_minutes = pause["duration"] / 60.0
        print(
            f"{idx:2d}. {duration_minutes:.2f} min | "
            f"start {pause['start_time']}, end {pause['end_time']} | "
            f"{pause['annotation']}"
        )


def main(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Report gaps between actions in processed trajectories.")
    parser.add_argument(
        "--data",
        help="Path to processed_trajectory.json file.",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=0.0,
        help="Only include gaps larger than this many seconds (default: 0.0).",
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        help="Path to timeline JSON file to merge pauses with.",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default=None,
        help="Output path for merged timeline JSON (default: timeline path with _with_pauses suffix).",
    )

    parsed = parser.parse_args(args)

    nodes = load_nodes(parsed.data)
    print(f"Loaded {len(nodes)} nodes from {parsed.data}")

    # Compute pauses (ctrl+c followed by gap > 2 minutes)
    pauses = compute_pauses(nodes, min_gap_minutes=2.0)
    print(f"Detected {len(pauses)} pauses (ctrl+c followed by gap > 2.0 minutes).")
    print_pause_summary(pauses)
    
    # Compute all large gaps (> 3 minutes) between actions
    large_gaps = compute_large_gaps(nodes, min_gap_minutes=3.0)
    print(f"\nDetected {len(large_gaps)} large gaps (> 3.0 minutes) between actions.")
    if large_gaps:
        sorted_gaps = sorted(large_gaps, key=lambda p: p["duration"], reverse=True)
        for idx, gap in enumerate(sorted_gaps[:10], 1):  # Show top 10
            duration_minutes = gap["duration"] / 60.0
            print(
                f"  {idx:2d}. {duration_minutes:.2f} min | "
                f"start {gap['start_time']:.2f}, end {gap['end_time']:.2f} | "
                f"{gap['annotation']}"
            )
        if len(large_gaps) > 10:
            print(f"  ... and {len(large_gaps) - 10} more gaps")

    # Organize outputs into subdirectories
    # Find the issue directory (walk up past processing step directories)
    base_dir = Path(parsed.data).parent
    current = base_dir
    while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified"]:
        current = current.parent
    output_base = current / "3_timeline" / "3_pauses"
    output_base.mkdir(parents=True, exist_ok=True)
    pauses_path = output_base / "pauses.json"
    save_gap_report(pauses, str(pauses_path))
    
    # Save large gaps separately
    if large_gaps:
        large_gaps_path = output_base / "large_gaps.json"
        save_gap_report(large_gaps, str(large_gaps_path))
    
    # Merge with timeline if provided
    if parsed.timeline:
        timeline_path = Path(parsed.timeline)
        print(f"\nLoading timeline from: {timeline_path}")
        timeline = load_json(timeline_path)
        print(f"  Loaded {len(timeline)} timeline segments")
        
        # Print original timeline breakdown
        print_timeline_statistics(timeline, "Original Timeline Activity Breakdown")
        
        # Check for overlaps with pauses and large gaps (informational only)
        if pauses:
            print("\n--- Checking overlaps with pauses ---")
            check_overlaps(timeline, pauses)
        
        # Remove large gaps that are already in pauses (to avoid duplicates)
        large_gaps_unique = deduplicate_gaps(pauses, large_gaps)
        if len(large_gaps_unique) < len(large_gaps):
            print(f"\nFiltered out {len(large_gaps) - len(large_gaps_unique)} large gaps that are already pauses.")
        
        if large_gaps_unique:
            print("\n--- Checking overlaps with large gaps ---")
            check_overlaps(timeline, large_gaps_unique)
        
        # Split timeline segments that overlap with gaps
        all_gaps_to_add = pauses + large_gaps_unique
        timeline_split = split_timeline_by_gaps(timeline, all_gaps_to_add)
        if len(timeline_split) != len(timeline):
            print(f"\nSplit timeline segments: {len(timeline)} -> {len(timeline_split)} segments")
        
        # Merge split timeline with gaps
        merged = merge_timeline_and_pauses(timeline_split, all_gaps_to_add)
        print(f"\nMerged timeline contains {len(merged)} segments (split timeline: {len(timeline_split)}, pauses: {len(pauses)}, large gaps: {len(large_gaps_unique)})")
        
        # Print merged timeline breakdown
        print_timeline_statistics(merged, "Merged Timeline Activity Breakdown")
        
        # Calculate output_base from timeline path (should be in 3_timeline/2_gaps/)
        # Find the issue directory (walk up past processing step directories)
        current = timeline_path.parent
        while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified"]:
            current = current.parent
        # Create path: issue_dir/3_timeline/2_gaps/3_pauses
        merge_output_base = current / "3_timeline" / "3_pauses"
        merge_output_base.mkdir(parents=True, exist_ok=True)
        
        # Save merged timeline
        if parsed.merge_output:
            merge_output_path = Path(parsed.merge_output)
        else:
            merge_output_path = merge_output_base / "with_pauses.json"
        
        save_json(merged, merge_output_path)
        print(f"Saved merged timeline to: {merge_output_path}")


if __name__ == "__main__":
    main()
