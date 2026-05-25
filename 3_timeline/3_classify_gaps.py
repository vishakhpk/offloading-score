"""Classify remaining gaps between timeline activities using LLM."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

# Ensure project root is importable for utils/language
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    pstr = str(path)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

from utils import call_openai, load_json, save_json

from dotenv import load_dotenv
load_dotenv()

GAP_CLASSIFY_PROMPT = """You are labeling time gaps between developer actions.

A gap represents a period where no recorded actions occurred. 

You will be given:
- The label and annotation of the activity that occurred BEFORE the gap
- The label and annotation of the activity that occurred AFTER the gap
- The duration of the gap

Assign a label to this gap. The label should be:
- The BEFORE activity's label if the gap is part of continuing the previous activity
- The AFTER activity's label if the gap is part of preparing for the next activity
- One of these gap labels if the gap represents a distinct new activity:
  - "compiling": waiting for code to compile
  - "waiting on generation": waiting for AI to generate response
  - "pause": away from keyboard
  - "unknown": unable to classify

Respond with: label, justification

Format: label, justification
"""


def parse_classification(response: str) -> dict[str, Any]:
    """Parse LLM classification response.
    
    Expected format: label, justification
    """
    parts = [p.strip() for p in response.split(",")]
    
    if len(parts) < 2:
        # Try to parse more flexibly
        label = parts[0] if len(parts) > 0 and parts[0] != "N/A" else None
        justification = ", ".join(parts[1:]) if len(parts) > 1 else "Unable to parse response"
    else:
        label = parts[0] if parts[0] != "N/A" else None
        justification = ", ".join(parts[1:])
    
    return {
        "label": label,
        "justification": justification
    }


def classify_gap_with_llm(gap: dict[str, Any], model_name: str = "gpt-5.1", verbose: bool = False) -> dict[str, Any]:
    """Classify a gap using LLM.
    
    Args:
        gap: Gap dictionary with duration and context
        model_name: LLM model name
        verbose: Print detailed output
    """
    before_label = gap["before_activity"]["label"]
    after_label = gap["after_activity"]["label"]
    before_ann = gap["before_activity"].get("annotation", "")
    after_ann = gap["after_activity"].get("annotation", "")
    duration = gap["duration"]
    duration_min = duration / 60.0
    
    # Build context for LLM with labels and annotations
    context = f"""Gap Details:
- Duration: {duration:.2f} seconds ({duration_min:.2f} minutes)

Before activity:
- Label: {before_label}
- Annotation: {before_ann if before_ann else "No annotation"}

After activity:
- Label: {after_label}
- Annotation: {after_ann if after_ann else "No annotation"}

Time range: {datetime.fromtimestamp(gap['start_time']).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(gap['end_time']).strftime('%Y-%m-%d %H:%M:%S')}

Assign a label to this gap. Use "{before_label}" if continuing the previous activity, "{after_label}" if preparing for the next activity, or one of: compiling, waiting on AI generation, pause, unknown."""
    
    if verbose:
        print(f"  Classifying gap of {duration:.2f}s ({duration_min:.2f} min) between '{before_label}' and '{after_label}'...")
    
    response = call_openai(GAP_CLASSIFY_PROMPT, context, model_name=model_name)
    
    if not response:
        # Fallback classification
        gap["label"] = "unknown"
        gap["justification"] = "LLM classification failed, using default"
        return gap
    
    # Parse response
    parsed = parse_classification(response)
    
    gap["label"] = parsed["label"]
    gap["justification"] = parsed["justification"]
    
    if verbose:
        print(f"  Result: label: {gap.get('label', 'N/A')}")
        print(f"  Justification: {gap['justification']}")
    
    return gap


def classify_gaps_batch(gaps: list[dict[str, Any]], model_name: str = "gpt-4o", verbose: bool = True) -> list[dict[str, Any]]:
    """Classify all gaps using LLM."""
    classified_gaps = []
    
    for i, gap in enumerate(gaps):
        if verbose:
            print(f"\nClassifying gap {i+1}/{len(gaps)}...")
        
        classified_gap = classify_gap_with_llm(gap, model_name=model_name, verbose=verbose)
        classified_gaps.append(classified_gap)
    
    return classified_gaps


def find_gaps_in_timeline(timeline: List[Dict[str, Any]], min_gap_seconds: float = 0.0) -> List[Dict[str, Any]]:
    """Find gaps between timeline entries.
    
    Args:
        timeline: List of timeline entries sorted by start_time
        min_gap_seconds: Minimum gap duration to consider (default: 0.0 seconds)
    
    Returns:
        List of gap entries with start, end, duration, and context
    """
    gaps = []
    
    # Sort timeline by start_time if not already sorted
    sorted_timeline = sorted(timeline, key=lambda x: x["start_time"])
    
    for i in range(len(sorted_timeline) - 1):
        current = sorted_timeline[i]
        next_entry = sorted_timeline[i + 1]
        
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


def convert_gaps_to_timeline_entries(classified_gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert classified gaps into timeline entries.
    
    Args:
        classified_gaps: List of classified gap dictionaries
    
    Returns:
        List of timeline entries with start_time, end_time, duration, label, annotation
    """
    timeline_entries = []
    
    for gap in classified_gaps:
        entry = {
            "start_time": gap["start_time"],
            "end_time": gap["end_time"],
            "duration": gap["duration"],
            "label": gap.get("label", "unknown"),
            "annotation": gap.get("justification", "")
        }
        timeline_entries.append(entry)
    
    return timeline_entries


def merge_timeline_with_gaps(timeline: list[dict[str, Any]], gap_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge timeline entries with gap entries and sort by start_time.
    
    Args:
        timeline: Original timeline entries
        gap_entries: Gap entries converted to timeline format
    
    Returns:
        Merged and sorted timeline
    """
    merged = timeline + gap_entries
    return sorted(merged, key=lambda x: x["start_time"])


def print_classification_summary(classified_gaps: list[dict[str, Any]]) -> None:
    """Print summary of gap classifications."""
    from collections import Counter
    
    print("\n--- Classification Summary ---")
    
    # Count by label
    label_counts = Counter(g.get("label") for g in classified_gaps if g.get("label"))
    if label_counts:
        print("Labels assigned to gaps:")
        for label, count in label_counts.most_common():
            total_duration = sum(g["duration"] for g in classified_gaps if g.get("label") == label)
            print(f"  {label}: {count} gaps, {total_duration:.2f}s ({total_duration/60:.2f} min)")
    


def main():
    parser = argparse.ArgumentParser(
        description="Classify remaining gaps between timeline activities using LLM."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to timeline JSON file or remaining gaps JSON file."
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=5.0,
        help="Minimum gap duration in seconds to classify (default: 5.0)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress."
    )
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    data = load_json(data_path)
    
    # Check if it's a timeline (list of entries with start_time/end_time) or gaps (list with gap_index)
    if not data:
        print("No data found to process.")
        return
    
    # Determine if it's a timeline or gaps file
    first_item = data[0] if isinstance(data, list) else data
    is_timeline = "start_time" in first_item and "end_time" in first_item and "label" in first_item
    is_gaps = "gap_index" in first_item and "before_activity" in first_item
    
    original_timeline = None
    gaps_to_classify = []
    
    if is_timeline:
        print(f"Loaded timeline with {len(data)} entries from {data_path}")
        original_timeline = data
        
        # Find gaps between timeline entries
        gaps_between = find_gaps_in_timeline(data, min_gap_seconds=args.min_gap)
        print(f"Found {len(gaps_between)} gaps between timeline entries")
        gaps_to_classify.extend(gaps_between)
        
        # Find segments labeled as "gap" or "unknown"
        gap_segments = [entry for entry in data if entry.get("label") in ["gap", "unknown"]]
        print(f"Found {len(gap_segments)} segments labeled as 'gap' or 'unknown'")
        
        # Convert gap/unknown segments to gap format for classification
        for i, segment in enumerate(gap_segments):
            # Find before and after activities
            timeline_sorted = sorted(data, key=lambda x: x["start_time"])
            segment_idx = timeline_sorted.index(segment)
            
            before_activity = None
            after_activity = None
            
            if segment_idx > 0:
                before_entry = timeline_sorted[segment_idx - 1]
                before_activity = {
                    "label": before_entry.get("label", "unknown"),
                    "annotation": before_entry.get("annotation", ""),
                    "end_time": before_entry.get("end_time")
                }
            
            if segment_idx < len(timeline_sorted) - 1:
                after_entry = timeline_sorted[segment_idx + 1]
                after_activity = {
                    "label": after_entry.get("label", "unknown"),
                    "annotation": after_entry.get("annotation", ""),
                    "start_time": after_entry.get("start_time")
                }
            
            # Create gap entry from segment
            gap_entry = {
                "gap_index": f"segment_{i}",
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment.get("duration", segment["end_time"] - segment["start_time"]),
                "before_activity": before_activity or {
                    "label": "unknown",
                    "annotation": "No previous activity",
                    "end_time": segment["start_time"]
                },
                "after_activity": after_activity or {
                    "label": "unknown",
                    "annotation": "No next activity",
                    "start_time": segment["end_time"]
                },
                "is_segment": True,  # Mark that this came from a segment, not a gap between entries
                "original_label": segment.get("label")
            }
            gaps_to_classify.append(gap_entry)
        
        if not gaps_to_classify:
            print("No gaps or gap/unknown segments found to classify.")
            return
        print(f"Total items to classify: {len(gaps_to_classify)}")
        
    elif is_gaps:
        print(f"Loaded {len(data)} gaps from {data_path}")
        gaps_to_classify = data
    else:
        raise ValueError(f"Unknown data format in {data_path}. Expected timeline or gaps file.")
    
    gaps = gaps_to_classify
    
    if not gaps:
        print("No gaps or gap/unknown segments found to classify.")
        return
    
    total_gap_time = sum(g["duration"] for g in gaps)
    print(f"Total gap time: {total_gap_time:.2f}s ({total_gap_time/60:.2f} min)")
    
    # Classify gaps
    print(f"\nClassifying gaps using gpt-5.1...")
    classified_gaps = classify_gaps_batch(gaps, model_name="gpt-5.1", verbose=args.verbose)
    
    # Print summary
    print_classification_summary(classified_gaps)
    
    # Organize outputs into subdirectories
    # Find the issue directory (walk up past processing step directories)
    current = data_path.parent
    while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified"]:
        current = current.parent
    output_base = current / "3_timeline" / "4_classified"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Convert gaps to timeline entries and merge with original timeline
    print(f"\nMerging classified gaps into timeline...")
    gap_entries = convert_gaps_to_timeline_entries(classified_gaps)
    
    # If we had an original timeline, replace gap/unknown segments with classified versions
    if original_timeline:
        # Remove original gap/unknown segments and add classified versions
        filtered_timeline = [entry for entry in original_timeline if entry.get("label") not in ["gap", "unknown"]]
        merged_timeline = merge_timeline_with_gaps(filtered_timeline, gap_entries)
    else:
        merged_timeline = gap_entries
    
    # Save merged timeline (this is the main output)
    timeline_output_path = output_base / "classified.json"
    save_json(merged_timeline, timeline_output_path)
    print(f"\nSaved merged timeline with classified gaps to: {timeline_output_path}")
    
    # Also save just the classified gaps for reference
    gaps_output_path = output_base / "classified_gaps.json"
    save_json(classified_gaps, gaps_output_path)
    print(f"Saved classified gaps (gaps only) to: {gaps_output_path}")
    
    # Print detailed results
    if args.verbose:
        print("\n--- Detailed Gap Classifications ---")
        for i, gap in enumerate(classified_gaps, 1):
            print(f"\nGap {i}:")
            print(f"  Duration: {gap['duration']:.2f}s ({gap['duration']/60:.2f} min)")
            print(f"  Between: '{gap['before_activity']['label']}' -> '{gap['after_activity']['label']}'")
            if gap.get('label'):
                print(f"  Label: {gap['label']}")
            print(f"  Justification: {gap.get('justification', 'N/A')}")


if __name__ == "__main__":
    main()

