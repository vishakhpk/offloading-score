"""
Post-processing functions for grouped segments.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable for utils/language
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    pstr = str(path)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

from language import get_first_action, get_last_action, merge_nodes
from utils import call_openai, extract_segments, load_json, save_json, save_segments

# Import load_labels from 0_induce.py
import importlib.util
induce_path = Path(__file__).parent / "0_induce.py"
spec = importlib.util.spec_from_file_location("induce", induce_path)
induce_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(induce_module)
load_labels = induce_module.load_labels

from dotenv import load_dotenv
load_dotenv()


def merge_adjacent_same_label(segments: list[dict[str, Any]], verbose: bool = False) -> list[dict[str, Any]]:
    """
    Merge adjacent segments that have the same label.
    
    Args:
        segments: List of segment dictionaries with labels
        verbose: Print progress information
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    merged = []
    current_segment = segments[0].copy()
    current_label = current_segment.get("label")
    current_source_indices = set(current_segment.get("source_indices", []))
    
    for i in range(1, len(segments)):
        next_segment = segments[i]
        next_label = next_segment.get("label")
        
        # If same label, merge them
        if current_label == next_label:
            # Merge nodes using helper function
            merged_node = merge_nodes([current_segment, next_segment])
            
            # Preserve and merge metadata
            merged_node["label"] = current_label
            
            # Merge source indices
            next_source_indices = set(next_segment.get("source_indices", []))
            current_source_indices.update(next_source_indices)
            merged_node["source_indices"] = sorted(list(current_source_indices))
            
            # Update annotation (combine if different)
            current_annotation = current_segment.get("annotation", "")
            next_annotation = next_segment.get("annotation", "")
            if current_annotation != next_annotation:
                merged_node["annotation"] = f"{current_annotation}; {next_annotation}"
            else:
                merged_node["annotation"] = current_annotation
            
            # Update time range using helper functions
            first_action = get_first_action(merged_node)
            last_action = get_last_action(merged_node)
            
            if first_action and last_action:
                first_time = first_action.get("time", {}).get("before")
                last_time = last_action.get("time", {}).get("after")
                if first_time and last_time:
                    merged_node["time"] = {
                        "start": first_time,
                        "end": last_time,
                    }
            
            current_segment = merged_node
            
            if verbose:
                print(f"Merging adjacent segments with label '{current_label}'")
        else:
            # Different label, save current and start new
            merged.append(current_segment)
            current_segment = next_segment.copy()
            current_label = next_label
            current_source_indices = set(current_segment.get("source_indices", []))
    
    # Add the last segment
    merged.append(current_segment)
    
    if verbose:
        print(f"Merged {len(segments)} segments into {len(merged)} segments")
    
    return merged


UNKNOWN_CLASSIFY_PROMPT = """You are classifying a coding activity that was previously labeled as "unknown".

You will be given:
- The annotation/description of the activity that needs classification
- The label and annotation of the activity that occurred BEFORE it
- The label and annotation of the activity that occurred AFTER it
- A list of available activity labels

Based on the context (what happened before and after), classify the unknown activity with the most appropriate label from the available labels and provide a description similar in style to the annotations provided.

Respond in the format:
(label) Description

Example responses:
(writing code) Edits the main function to add error handling"""


def reclassify_unknown_labels(
    segments: list[dict[str, Any]],
    labels: list[str],
    model_name: str = "gpt-5.1",
    verbose: bool = False
) -> list[dict[str, Any]]:
    """Reclassify segments with "unknown" label based on before/after activities."""
    labels_text = "\n".join([f"- {label}" for label in labels])
    reclassified_count = 0
    
    for i, segment in enumerate(segments):
        if segment.get("label") != "unknown":
            continue
        
        before = segments[i - 1] if i > 0 else None
        after = segments[i + 1] if i < len(segments) - 1 else None
        annotation = segment.get("annotation", "No annotation available")
        
        # Build context
        context = f"Activity to classify: {annotation}\n\nAvailable labels:\n{labels_text}\n"
        if before:
            context += f"\nBefore: ({before.get('label', 'unknown')}) {before.get('annotation', '')}\n"
        if after:
            context += f"After: ({after.get('label', 'unknown')}) {after.get('annotation', '')}\n"
        
        if verbose:
            print(f"Reclassifying: {annotation[:50]}...")
        
        response = call_openai(UNKNOWN_CLASSIFY_PROMPT, context, model_name=model_name)
        if not response:
            continue
        
        # Parse (label) Description format
        match = re.match(r'\(([^)]+)\)\s*(.+)', response.strip())
        if match:
            new_label, new_description = match.group(1).strip(), match.group(2).strip()
            if new_label in labels:
                segment["label"] = new_label
                segment["annotation"] = new_description
                reclassified_count += 1
                if verbose:
                    print(f"  → {new_label}: {new_description}")
    
    if verbose:
        print(f"Reclassified {reclassified_count} unknown segment(s)")
    
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Post-process grouped segments by merging adjacent segments with the same label."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to grouped segments JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path (defaults to <input> with _merged suffix)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress."
    )
    parser.add_argument(
        "--reclassify-unknown",
        action="store_true",
        help="Reclassify segments with 'unknown' label using LLM."
    )
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    data = load_json(data_path)
    segments = extract_segments(data)
    
    if not segments:
        raise ValueError(f"No segments found in {data_path}")
    
    if args.verbose:
        print(f"Processing {len(segments)} segments...")
    
    # Merge adjacent segments with same label
    merged_segments = merge_adjacent_same_label(segments, verbose=args.verbose)

    
    # Organize outputs into subdirectories
    # Find the issue directory (parent of 1_segment or 0_preprocessing, etc.)
    # Then use 2_induction within that issue directory
    current = data_path.parent
    # Walk up until we find a directory that's not a processing step directory
    while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline"]:
        current = current.parent
    # Now current is the issue directory, create 2_induction there
    output_base = current / "2_induction"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_base / "merged.json"
    
    save_json(merged_segments, output_path)
    print(f"\nSaved {len(merged_segments)} merged segments to: {output_path}")
    
    # Save individual segment files
    output_dir = output_base / "merged"
    save_segments(
        merged_segments,
        output_file=None,  # Already saved above
        output_dir=output_dir,
    )
    print(f"Saved {len(merged_segments)} segment files to: {output_dir}/")
    
    # Generate and save updated workflow.txt
    workflow_lines = []
    for i, segment in enumerate(merged_segments):
        label = segment.get("label", "unknown")
        annotation = segment.get("annotation", "")
        source_indices = segment.get("source_indices", [])
        
        if source_indices:
            start_idx = min(source_indices)
            end_idx = max(source_indices)
            if start_idx == end_idx:
                range_str = f"[{start_idx}-{start_idx}]"
            else:
                range_str = f"[{start_idx}-{end_idx}]"
        else:
            range_str = f"[{i}-{i}]"
        
        workflow_lines.append(f"{range_str} ({label}) {annotation}")
    
    workflow_text = "\n".join(workflow_lines)
    workflow_path = output_base / "workflow.txt"
    with open(workflow_path, "w") as f:
        f.write(workflow_text)
    print(f"\nSaved updated workflow to: {workflow_path}")


if __name__ == "__main__":
    main()

