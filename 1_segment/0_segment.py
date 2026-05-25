"""
Segment a processed trajectory into raw nodes based on visual similarity.

- Measure state similarity with image MSE and split when the diff exceeds a threshold.
- Optionally re-merge short segments when an LLM deems adjacent software identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import call_openai, encode_image_for_llm, load_image, load_json, resolve_image_path, save_json, save_segments  # type: ignore
from language import (  # type: ignore
    clone_node,
    get_first_action,
    get_last_action,
    merge_nodes,
    node_length,
    state_path,
    wrap_sequence,
)

from dotenv import load_dotenv
load_dotenv()

# Constants
MAX_DIFF = 10_0000.0
REMERGE_SIM_THRESHOLD = 0.8  # min probability (0-1) that two screens are same session

# A subaction typically involves:
# - Working on the same specific code element (same function, same class, same file section)
# - Performing a related operation in the same context (e.g., making edits to the same function, running the same test, viewing the same documentation)
# - Continuing the same immediate objective without switching to a different code area or task


NEURAL_PROMPT = """You are given two screenshots from a coding session. Your task is to determine if these two screenshots represent the same subaction.

A subaction is a small, modular unit of work - a single step or operation that the developer is performing. 
Examples: writing/editing code in a specific function, running tests or comands, navigating files

What to check:
- Same file and code content? (same function, class, method, code block)
- Same activity type? (both writing, both reading, both running)
- Same tool/context? (same IDE tab, same terminal, same browser tab)
- Continuity? (continuing the same immediate work vs switching operations)

Compare the two screenshots:
- If they represent the same subaction (same code element, same context, continuation of the same immediate work), return 1.0
- If they represent different subactions (different code elements, different context, switched to a different operation), return 0.0
- For intermediate cases where the relationship is unclear, return a value between 0 and 1 based on how likely they are to be the same subaction

Format your response as:
SCORE: <number between 0 and 1>
JUSTIFICATION: <brief explanation of why this score was chosen, focusing on whether they represent the same subaction>"""

# ---------------------------------------------------------------------------
# Image helpers


def _compute_mse(img1: Image.Image, img2: Image.Image) -> float:
    diff = ImageChops.difference(img1, img2)
    stats = ImageStat.Stat(diff)
    return float(sum(value ** 2 for value in stats.rms) / len(stats.rms))


def mse(image_path1: str | None, image_path2: str | None) -> float:
    """Mean squared error between two screenshots. Resizes images to match if sizes differ."""
    if image_path1 is None or image_path2 is None:
        print("Missing image paths")
        return MAX_DIFF
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)
    if img1 is None or img2 is None:
        print("Invalid images: failed to load")
        return MAX_DIFF
    
    # Resize images to match if sizes differ (resize both to smaller dimensions to minimize distortion)
    if img1.size != img2.size:
        # Use the smaller dimensions to avoid upscaling artifacts
        target_size = (
            min(img1.size[0], img2.size[0]),
            min(img1.size[1], img2.size[1])
        )
        if img1.size != target_size:
            img1 = img1.resize(target_size, Image.Resampling.LANCZOS)
        if img2.size != target_size:
            img2 = img2.resize(target_size, Image.Resampling.LANCZOS)
    
    return _compute_mse(img1, img2)


def neural(image1: str | None, image2: str | None) -> tuple[float, str]:
    """
    LLM-based similarity: returns (probability, justification) tuple.
    Probability is 0.0-1.0 that the two screens are the same session. Higher = more similar.
    """
    if image1 is None or image2 is None:
        return 0.0, "Missing image paths"
    path1 = resolve_image_path(image1)
    path2 = resolve_image_path(image2)
    if path1 is None or path2 is None:
        return 0.0, "Image paths not found"
    
    # Encode images for LLM API
    encoded1 = encode_image_for_llm(path1)
    encoded2 = encode_image_for_llm(path2)
    if encoded1 is None or encoded2 is None:
        return 0.0, "Failed to encode images"
    
    content = [encoded1, encoded2]
    
    try:
        response = call_openai(prompt=NEURAL_PROMPT, content=content)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] call_openai failed: {exc}")
        return 0.0, f"Error calling LLM: {exc}"

    # Parse score and justification from response
    if isinstance(response, str):
        import re
        response_clean = response.strip()
        
        # Extract score: try "SCORE: X" format first, then any decimal number
        score_match = re.search(r"SCORE:\s*(\d+\.?\d*)", response_clean, re.IGNORECASE)
        if not score_match:
            score_match = re.search(r"(\d+\.?\d+)", response_clean)
        
        if score_match:
            try:
                val = float(score_match.group(1))
                score = max(0.0, min(1.0, val))
            except ValueError:
                print(f"[WARNING] Could not parse score from response: {response_clean[:50]}")
                return 0.0, f"Could not parse score from response"
        else:
            print(f"[WARNING] Could not find score in response: {response_clean[:50]}")
            return 0.0, f"Could not parse score from response"
        
        # Extract justification from "JUSTIFICATION: ..." format (capture everything after until end)
        just_match = re.search(r"JUSTIFICATION:\s*(.+)$", response_clean, re.IGNORECASE | re.DOTALL)
        if just_match:
            justification = just_match.group(1).strip()
        else:
            # Fallback: use the full response as justification if no explicit field
            justification = response_clean
        
        return score, justification
    
    return 0.0, "Invalid response type"


# ---------------------------------------------------------------------------
# Trajectory utilities

def annotate_state_diffs(nodes: list[dict[str, Any]], verbose: bool = False) -> list[float]:
    scores: list[float] = []
    if not nodes:
        return scores
    nodes[0].setdefault("state", {})["diff_score"] = 0.0
    prev = nodes[0]
    for idx in range(1, len(nodes)):
        curr = nodes[idx]
        curr_state = curr.setdefault("state", {})
        diff = mse(state_path(prev, reverse=True), state_path(curr, reverse=True))
        curr_state["diff_score"] = diff
        scores.append(diff)
        prev = curr
        if verbose:
            print(f"[STATE] step {idx}: {diff:.2f}")
    return scores


def annotate_transition_diffs(nodes: list[dict[str, Any]], verbose: bool = False) -> list[float]:
    transitions: list[float] = []
    if not nodes:
        return transitions
    nodes[0].setdefault("state", {})["transition_diff"] = 0.0
    prev_path = state_path(nodes[0], reverse=False)
    for idx in range(1, len(nodes)):
        curr = nodes[idx]
        curr_state = curr.setdefault("state", {})
        curr_path = state_path(curr, reverse=False)
        if prev_path and curr_path:
            diff = mse(prev_path, curr_path)
        else:
            print("Missing image paths for transition diff")
            diff = MAX_DIFF
        curr_state["transition_diff"] = diff
        transitions.append(diff)
        prev_path = curr_path
        if verbose:
            print(f"[TRANSITION] step {idx}: {diff:.2f}")
    return transitions


def get_metric_value(node: dict[str, Any]) -> float:
    """Get the transition_diff value."""
    state = node.get("state") or {}
    transition_val = state.get("transition_diff")
    transition_val = float(transition_val) if isinstance(transition_val, (int, float)) else MAX_DIFF
    return transition_val

def segment_per_step(
    nodes: list[dict[str, Any]],
    threshold: float,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for node in nodes:
        diff = get_metric_value(node)
        if current and diff > threshold:
            segments.append(wrap_sequence(current))
            current = []
        current.append(clone_node(node))
    if current:
        segments.append(wrap_sequence(current))
    if verbose:
        lengths = [node_length(seg) for seg in segments]
        print(f"[SEGMENTS] {len(segments)} segments via point split: {lengths}")
    return segments


def get_state_similarity_neural(
    curr_node: dict[str, Any],
    prev_node: dict[str, Any],
) -> tuple[float, str]:
    """Get neural similarity between two nodes. Returns (score, justification) tuple."""
    curr_action = get_first_action(curr_node)
    prev_action = get_last_action(prev_node)
    curr_path = state_path(curr_action or {}, reverse=True)
    prev_path = state_path(prev_action or {}, reverse=True)
    return neural(prev_path, curr_path)


def print_similarity_distribution(similarities: list[float], label: str = "Neural similarity") -> None:
    """Print distribution statistics for similarity scores."""
    if not similarities:
        return
    arr = np.array(similarities, dtype=np.float64)
    print(f"[{label}] Distribution: n={len(similarities)}")
    print(f"  min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}, median={np.median(arr):.3f}")
    print(f"  p25={np.percentile(arr, 25):.3f}, p75={np.percentile(arr, 75):.3f}, p90={np.percentile(arr, 90):.3f}, p95={np.percentile(arr, 95):.3f}")


def print_mse_distribution(
    scores: list[float],
    threshold: float | None = None,
    label: str = "MSE scores",
    metadata_lines: list[str] | None = None,
) -> None:
    """Print distribution statistics for MSE scores with threshold analysis."""
    if not scores:
        return
    arr = np.array(scores, dtype=np.float64)
    lines = [
        f"[{label}] Distribution: n={len(scores)}",
        f"  min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}",
        f"  p25={np.percentile(arr, 25):.2f}, p50={np.percentile(arr, 50):.2f}, p75={np.percentile(arr, 75):.2f}",
        f"  p90={np.percentile(arr, 90):.2f}, p95={np.percentile(arr, 95):.2f}, p99={np.percentile(arr, 99):.2f}",
    ]
    
    if threshold is not None:
        above = sum(arr > threshold)
        lines.extend([
            f"",
            f"  Threshold: {threshold:.2f}",
            f"  Scores above threshold: {above}/{len(scores)} ({100*above/len(scores):.1f}%)",
            f"  This would create ~{above + 1} segments (assuming splits at each threshold crossing)",
        ])
    
    # Print to console
    print()  # Add newline before distribution
    for line in lines:
        print(line)
    
    # Save to metadata if provided
    if metadata_lines is not None:
        metadata_lines.extend(lines)
        

def remerge_segments(
    segments: list[dict[str, Any]],
    sim_threshold: float,
    verbose: bool = False,
    collect_similarities: bool = False,
) -> tuple[list[dict[str, Any]], bool, list[float]]:
    if len(segments) <= 1:
        return segments, False, []
    merged: list[dict[str, Any]] = []
    idx = 0
    changed = False
    similarities = [] if collect_similarities else None
    while idx < len(segments) - 1:
        curr = segments[idx]
        nxt = segments[idx + 1]
        prob_same, justification = get_state_similarity_neural(nxt, curr)
        
        if collect_similarities:
            similarities.append(prob_same)
        
        # Save similarity score and justification in the next segment's metadata (even if not merged)
        nxt_state = nxt.setdefault("state", {})
        nxt_state["neural_similarity_to_prev"] = prob_same
        nxt_state["neural_similarity_justification"] = justification
        
        if prob_same >= sim_threshold:
            merged_seg = merge_nodes([curr, nxt])
            # Preserve all state information from nxt (including similarity metadata)
            merged_state = merged_seg.setdefault("state", {})
            nxt_state = nxt.get("state", {})
            # Copy all state from nxt to merged segment
            merged_state.update(nxt_state)
            # Ensure similarity metadata is set (in case nxt didn't have state)
            merged_state["neural_similarity_to_prev"] = prob_same
            merged_state["neural_similarity_justification"] = justification
            merged.append(merged_seg)
            idx += 2
            changed = True
            if verbose:
                print(f"[REMERGE] merged segments {idx-1} and {idx} (p_same={prob_same:.2f})")
            continue
        # When not merging, curr is appended as-is, but it already has its metadata from when it was compared
        # to the previous segment (or was created from a merge). No need to update it here.
        merged.append(curr)
        idx += 1
    if idx == len(segments) - 1:
        merged.append(segments[idx])
    return merged, changed, similarities or []


def remerge_segments_iterative(
    segments: list[dict[str, Any]],
    sim_threshold: float,
    verbose: bool = False,
) -> list[dict[str, Any]]:

    pass_num = 0
    while pass_num < 1:
        num_segments_before = len(segments)
        segments, changed, similarities = remerge_segments(
            segments, sim_threshold, verbose, collect_similarities=True
        )
        num_segments_after = len(segments)
        label = "Neural similarity (pre-merge)" if pass_num == 0 else f"Neural similarity (after pass {pass_num})"
        print_similarity_distribution(similarities, label)
        print(f"  Threshold: {sim_threshold:.3f}")
        print(f"  Segments above threshold: {sum(s >= sim_threshold for s in similarities)}/{len(similarities)}")
        print(f"  Segments: {num_segments_before} -> {num_segments_after} (merged {num_segments_before - num_segments_after})")
        if verbose:
            print(f"[PASS {pass_num + 1}] Merged segments, changed: {changed}")
        if not changed:
            break
        pass_num += 1
    return segments


# ---------------------------------------------------------------------------
# IO helpers

def load_trajectory(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("node_type") != "sequence":
        raise ValueError("Trajectory root must be a sequence node.")
    return data


# ---------------------------------------------------------------------------
# Runner

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment a processed trajectory into raw nodes.")
    parser.add_argument("--trajectory", type=Path, required=True, help="Path to processed_trajectory.json.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="MSE threshold for splitting segments. If provided, overrides --auto-percentile.",
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
        default=REMERGE_SIM_THRESHOLD,
        help="Minimum probability (0-1) that adjacent segments are same session to merge.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug information.")
    return parser


def run_segmentation(args: argparse.Namespace) -> list[dict[str, Any]]:
    trajectory_path = args.trajectory.expanduser().resolve()
    root = load_trajectory(trajectory_path)
    nodes = root.get("nodes", [])
    if not nodes:
        print("Trajectory has no nodes.")
        return []

    # Annotate nodes with transition diff scores
    transition_scores = annotate_transition_diffs(nodes, args.verbose)

    # Determine threshold using transition scores only
    max_len = len(transition_scores)
    used_scores = transition_scores.copy()
    if not used_scores:
        raise SystemExit("Error: No scores computed. Need at least 2 nodes to compute differences.")
    
    # Collect metadata output
    metadata_lines: list[str] = []
    
    # Determine threshold: explicit value overrides percentile fallback
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "manual threshold"
    else:
        threshold = float(np.percentile(np.array(used_scores, dtype=np.float64), args.auto_percentile))
        threshold_source = f"{args.auto_percentile}th percentile"
    
    threshold_line = f"[THRESHOLD] Using {threshold_source}: {threshold:.2f}"
    print(f"\n{threshold_line}")
    metadata_lines.append(threshold_line)

    # Print detailed distribution analysis
    print_mse_distribution(used_scores, label="Transition diff scores", metadata_lines=metadata_lines)
    
    # Segment the trajectory (MSE-based)
    segments = segment_per_step(nodes, threshold, args.verbose)
    num_segments_before_neural = len(segments)

    # Organize outputs into subdirectories
    # If trajectory is in 0_preprocessing, output to issue root; otherwise use trajectory parent
    if trajectory_path.parent.name == "0_preprocessing":
        output_base = trajectory_path.parent.parent / "1_segment"
    else:
        output_base = trajectory_path.parent / "1_segment"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save MSE segments
    output_file_mse = output_base / "segments_mse.json"
    output_dir_mse = output_base / "segments_mse"
    save_segments(
        segments,
        output_file=output_file_mse.expanduser().resolve(),
        output_dir=output_dir_mse.expanduser().resolve(),
    )

    # Optionally re-merge short segments using LLM
    if args.do_remerge:
        segments = remerge_segments_iterative(
            segments,
            args.remerge_sim_threshold,
            args.verbose,
        )

    # Default outputs after neural remerge (or same as MSE if remerge disabled)
    output_file = output_base / "segments.json"
    output_dir = output_base / "segments"
    save_segments(
        segments,
        output_file=output_file.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
    )
    
    # Save metadata to file
    if metadata_lines:
        metadata_file = output_base / "segments_metadata.txt"
        with open(metadata_file, "w") as f:
            f.write("\n".join(metadata_lines))
        print(f"\n[SAVED] Metadata written to {metadata_file}")
    
    return segments


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_segmentation(args)


if __name__ == "__main__":
    main()
