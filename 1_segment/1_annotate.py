"""
Annotate action nodes with short descriptions of what is visible in the screenshots.

- Accepts a trajectory/segment JSON file (sequence root or list of nodes).
- For each action node, call an LLM with the before/after screenshots to summarize the user action.
- Writes the annotated structure back to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import call_openai, compose_key_input, encode_image_for_llm, extract_segments, get_key_input, is_keyboard_action, load_json, resolve_image_path, save_json, save_segments
from language import get_first_action, get_last_action, state_path

from dotenv import load_dotenv
load_dotenv()


SEQUENCE_PROMPT = """You are labeling coding workflow screenshots.
Given a sequence of actions with representative screenshots, write one sentence describing what the user is doing across this sequence.
The screenshots show key transitions in the sequence - the first screenshot shows the starting state, and subsequent screenshots show significant changes.
Call out the app/context (editor, terminal, browser, docs) and the overall intent or workflow (e.g., "User switches between editor and terminal, running tests and fixing errors", "User navigates through codebase searching for a specific function").
Example: "User switches from editor to terminal, runs `npm test`, then returns to editor to fix failing tests."
Respond with text only.

Context information:
- Issue context: If provided, this describes the overall task/goal the developer is working on. Use this to understand the broader purpose, but focus on describing the specific sequence visible in the screenshots.
- AI interactions: If provided, these show what the user asked AI or what AI responded during this sequence. This helps explain actions like sending prompts, applying AI suggestions, or reviewing AI output.

Guidelines:
- Treat any red box/crosshair as a locator only; describe the underlying control or content (tab, button, code line, chat input, etc.), not the box itself.
- Be specific about the app and visible context (files/tabs/dialogs/prompts). Expect IDEs/editors, terminals, browsers, docs, or chat panes.
- Focus on the specific UI component under the red box / crosshair and what the developer is trying to do now (navigate/open a file, run/test/debug, inspect/apply AI output, send a prompt/query, or edit a particular code section/line).
- Use the action type to guide emphasis: for clicks, name the target control; for scrolls, note what content is being reviewed; for keypress/text, note what is being edited or triggered.
- If AI interactions are present, incorporate them naturally (e.g., "User is sending a prompt to AI asking about...", "User is reviewing AI's response about...", "User is reviewing AI's suggestion to...").
- If issue context is provided, you can reference it briefly if relevant (e.g., "User is working on the requested feature by..."), but always prioritize describing what's visible in the screenshots.
- If only one screenshot is present (identical/low-diff pair), describe what is visible and most likely being done based on that single view.
- If unclear, give the most plausible interpretation without inventing unseen content."""


# ---------------------------------------------------------------------------
# Annotation

def _collect_all_actions(node: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Recursively collect all action nodes from a sequence node.
    Combines adjacent key press actions into single actions.
    """
    if node.get("node_type") == "action":
        return [node]
    actions = []
    for child in node.get("nodes", []):
        actions.extend(_collect_all_actions(child))
    
    # Combine adjacent key presses
    if not actions:
        return actions
    
    combined: list[dict[str, Any]] = []
    key_buffer: list[dict[str, Any]] = []
    
    def flush_key_buffer():
        """Combine buffered key presses into a single action."""
        nonlocal key_buffer, combined
        if not key_buffer:
            return
        if len(key_buffer) == 1:
            combined.append(key_buffer[0])
        else:
            # Extract key values and combine them
            key_values = [get_key_input(a.get("action", "")) for a in key_buffer]
            combined_keys = compose_key_input(key_values)
            if combined_keys:
                # Create a combined action node
                first_action = key_buffer[0]
                last_action = key_buffer[-1]
                combined_action = {
                    "node_type": "action",
                    "action": f"key_press('{combined_keys}')",
                    "state": {
                        "before": first_action.get("state", {}).get("before"),
                        "after": last_action.get("state", {}).get("after"),
                    },
                    "time": {
                        "before": first_action.get("time", {}).get("before"),
                        "after": last_action.get("time", {}).get("after"),
                    },
                }
                combined.append(combined_action)
            else:
                # If combination failed, add them individually
                combined.extend(key_buffer)
        key_buffer = []
    
    for action in actions:
        action_str = action.get("action", "")
        if is_keyboard_action(action_str):
            # Check if it's a simple key press (not a modifier or special key)
            key_value = get_key_input(action_str)
            # Only combine simple text keys (single characters, not special keys)
            if key_value and len(key_value) == 1 and not key_value.startswith("Key.") and "+" not in key_value:
                key_buffer.append(action)
            else:
                # Flush buffer and add this special key separately
                flush_key_buffer()
                combined.append(action)
        else:
            # Flush buffer and add this non-keyboard action
            flush_key_buffer()
            combined.append(action)
    
    flush_key_buffer()
    return combined


def _process_ai_events(ai_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Combine consecutive AI events of the same type.
    Especially useful for combining multiple LLM responses that aren't alternating with user requests.
    """
    if not ai_events:
        return []
    
    processed: list[dict[str, Any]] = []
    current_group: list[dict[str, Any]] = []
    current_type: str | None = None
    
    for event in ai_events:
        event_type = event.get("type", "unknown")
        
        # If same type as current group, add to group
        if event_type == current_type:
            current_group.append(event)
        else:
            # Flush current group if exists
            if current_group:
                combined = _combine_ai_events(current_group)
                processed.append(combined)
            
            # Start new group
            current_group = [event]
            current_type = event_type
    
    # Flush last group
    if current_group:
        combined = _combine_ai_events(current_group)
        processed.append(combined)
    
    return processed


def _combine_ai_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine multiple AI events of the same type into a single event."""
    if not events:
        return {}
    if len(events) == 1:
        return events[0]
    
    # Use metadata from first event
    combined = events[0].copy()
    
    # Combine content from all events
    contents = []
    for event in events:
        content = event.get("content", "")
        if content:
            contents.append(content)
    
    if contents:
        combined["content"] = "\n\n".join(contents)
    
    return combined


def _build_user_content(
    action_text: str,
    images: list[tuple[str, Path]] | None = None,
    ai_events: list[dict[str, Any]] | None = None,
    issue_description: str | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    
    # Add logged action text if available
    if action_text:
        content.append({"type": "text", "text": f"Logged action: {action_text}"})
    
    # Add issue description if available
    if issue_description:
        content.append({
            "type": "text",
            "text": f"Issue context: {issue_description}"
        })
    
    # Add AI context if available (process events first to combine consecutive ones)
    if ai_events:
        processed_events = _process_ai_events(ai_events)
        ai_context_parts = ["AI interactions during this action:"]
        for event in processed_events:
            event_type = event.get("type", "unknown")
            event_content = event.get("content", "")
            interface = event.get("interface", "")
            if event_content:
                context_line = f"- {event_type}"
                if interface:
                    context_line += f" ({interface})"
                context_line += f": {event_content[:200]}"  # Truncate long content
                if len(event_content) > 200:
                    context_line += "..."
                ai_context_parts.append(context_line)
        if len(ai_context_parts) > 1:  # More than just the header
            content.append({"type": "text", "text": "\n".join(ai_context_parts)})
    
    # Add all images (without labels)
    if images:
        for label, img_path in images:
            encoded = encode_image_for_llm(img_path)
            if encoded:
                content.append(encoded)
    
    return content


def _load_issue_description(trajectory_path: Path | None) -> str | None:
    """Load issue description from issue.json in the same directory as trajectory."""
    if trajectory_path is None:
        return None
    
    # Look for issue.json in the same directory as the trajectory file
    issue_json_path = trajectory_path.parent / "issue.json"
    if not issue_json_path.exists():
        return None
    
    try:
        issue_data = load_json(issue_json_path)
        # Try body_summary first, then summary, then body
        return issue_data.get("body_summary") or issue_data.get("summary") or issue_data.get("body")
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def annotate_action_node(
    node: dict[str, Any],
    *,
    screenshots_dir: Path | None,
    model_name: str,
    overwrite: bool = False,
    issue_description: str | None = None,
) -> bool:
    if node.get("node_type") != "action":
        return False
    if node.get("annotation") and not overwrite:
        return False

    state = node.get("state") or {}
    ai_events = node.get("ai_events")
    before = resolve_image_path(state.get("before"), screenshots_dir)
    after = resolve_image_path(state.get("after"), screenshots_dir)

    if before is None and after is None:
        node["annotation"] = "No screenshot available"
        return True

    # Build images list for action node
    images: list[tuple[str, Path]] = []
    if before:
        images.append(("Before screen", before))
    if after:
        images.append(("After screen", after))
    
    content = _build_user_content(
        node.get("action", ""),
        images=images if images else None,
        ai_events=ai_events,
        issue_description=issue_description,
    )
    try:
        response = call_openai(prompt=PROMPT, content=content, model_name=model_name)
    except Exception as exc:  # pragma: no cover - network failures
        print(f"[ERROR] call_openai failed for action: {exc}")
        return False

    if isinstance(response, str):
        description = response.strip()
    else:
        description = ""

    if not description:
        return False

    node["annotation"] = description
    return True


def _extract_sequence_info(
    sequence_node: dict[str, Any],
    *,
    screenshots_dir: Path | None,
    diff_threshold: float | None,
) -> dict[str, Any] | None:
    """
    Extract relevant information from a sequence node for LLM annotation.
    
    Returns:
        Dictionary with:
        - action_text: Combined action string
        - images: List of (label, path) tuples for screenshots with high transition_diff
        - ai_events: Combined AI events from all actions
        Returns None if sequence has no actions or screenshots
    """
    # Get all actions in the sequence (with key press combination)
    actions = _collect_all_actions(sequence_node)
    if not actions:
        return None

    # Collect all screenshots with high transition_diff
    images: list[tuple[str, Path]] = []
    
    # Always include before and after screenshots of the first and last actions
    first_action = get_first_action(sequence_node)
    last_action = get_last_action(sequence_node)
    
    if first_action:
        first_before_path = resolve_image_path(state_path(first_action, reverse=False), screenshots_dir)
        if first_before_path:
            images.append(("State 0 before (start)", first_before_path))

    # Include intermediate actions with high transition_diff (skip first and last)
    for i, action in enumerate(actions):
        # Skip first and last actions (already included above)
        if i == 0 or i == len(actions) - 1:
            continue
            
        state = action.get("state", {})
        transition_diff = state.get("transition_diff", 0.0)
        
        # Check if this action has a significant transition change
        transition_diff_val = float(transition_diff)
        
        if transition_diff_val >= diff_threshold:
            # Always include after screenshot if transition_diff is high
            after_path = resolve_image_path(state_path(action, reverse=False), screenshots_dir)
            images.append((f"State {i} before (diff={transition_diff_val:.0f})", after_path))
                # Always include after screenshot of the last action
    
    if last_action:
        last_after_path = resolve_image_path(state_path(last_action, reverse=True), screenshots_dir)
        if last_after_path:
            images.append(("State N after (end)", last_after_path))


    if not images:
        return None

    # Combine action strings
    action_strings = [a.get("action", "") for a in actions]
    combined_action = " + ".join(action_strings)

    # Collect AI events from all actions
    all_ai_events = []
    for action in actions:
        ai_events = action.get("ai_events")
        if ai_events:
            all_ai_events.extend(ai_events)

    return {
        "action_text": combined_action,
        "images": images,
        "ai_events": all_ai_events if all_ai_events else None,
    }


def annotate_sequence_node(
    node: dict[str, Any],
    *,
    screenshots_dir: Path | None,
    model_name: str,
    overwrite: bool = False,
    issue_description: str | None = None,
    transition_diff_threshold: float | None,
) -> bool:
    """Annotate a sequence node using first/last screenshots and high transition_diff intermediates."""
    if node.get("node_type") != "sequence":
        return False
    if node.get("annotation") and not overwrite:
        return False

    # Extract relevant information from the sequence node
    info = _extract_sequence_info(
        node,
        screenshots_dir=screenshots_dir,
        diff_threshold=transition_diff_threshold,
    )
    
    if info is None:
        node["annotation"] = "Sequence with no actions or screenshots with significant changes"
        return True

    # Build content for LLM
    content = _build_user_content(
        info["action_text"],
        images=info["images"],
        ai_events=info["ai_events"],
        issue_description=issue_description,
    )
    
    try:
        response = call_openai(prompt=SEQUENCE_PROMPT, content=content, model_name=model_name)
    except Exception as exc:  # pragma: no cover - network failures
        print(f"[ERROR] call_openai failed for sequence: {exc}")
        return False

    if isinstance(response, str):
        description = response.strip()
    else:
        description = ""

    if not description:
        return False

    node["annotation"] = description
    return True


def annotate_tree(
    root: Any,
    *,
    screenshots_dir: Path | None,
    model_name: str,
    overwrite: bool,
    trajectory_path: Path | None = None,
    transition_diff_threshold: float | None = None,
) -> dict[str, int]:
    stats = {"actions": 0, "annotated": 0, "skipped": 0}
    
    # Load issue description once for all actions
    issue_description = _load_issue_description(trajectory_path)

    # Get top-level sequence nodes
    if isinstance(root, list):
        top_level_sequences = root
    elif isinstance(root, dict) and root.get("node_type") == "sequence":
        top_level_sequences = root.get("nodes", [])
    else:
        top_level_sequences = []

    # Annotate each top-level sequence node (don't walk children)
    for sequence_node in top_level_sequences:
        if not isinstance(sequence_node, dict) or sequence_node.get("node_type") != "sequence":
            continue
        
        # Annotate the sequence node itself
        changed = annotate_sequence_node(
            sequence_node,
            screenshots_dir=screenshots_dir,
            model_name=model_name,
            overwrite=overwrite,
            issue_description=issue_description,
            transition_diff_threshold=transition_diff_threshold,
        )
        if changed:
            stats["annotated"] += 1
        else:
            stats["skipped"] += 1

    return stats


# ---------------------------------------------------------------------------
# IO helpers (using utils)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate each action node with a screenshot-based description."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to trajectory or segments JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (defaults to <input> with _annotated suffix).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run annotation even if an annotation already exists on the node.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=None,
        help="Minimum transition_diff or diff_score to include screenshots in sequence annotations. Should match the segmentation threshold used in 0_segment.py.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    # Output to the same directory as the input file
    output_base = input_path.parent
    output_base.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = output_base / "annotated.json"
    output_path = output_path.expanduser().resolve()

    # Use data/screenshots relative to input file's directory
    screenshots_dir = input_path.parent / "data" / "screenshots"

    data = load_json(input_path)
    
    # Try to get threshold from metadata.txt file if available
    threshold = args.diff_threshold
    if threshold is None:
        # Look for metadata.txt file (e.g., processed_trajectory_segments_metadata.txt)
        metadata_file = input_path.with_name(f"{input_path.stem}_metadata.txt")
        with open(metadata_file, "r") as f:
            for line in f:
                if line.startswith("[THRESHOLD] Using"):
                    # Parse: [THRESHOLD] Using 75.0th percentile: 1180.95
                    parts = line.split(":")
                    if len(parts) == 2:
                        threshold = float(parts[1].strip())
                        print(f"[THRESHOLD] Using threshold from metadata.txt: {threshold:.2f}")
                        break

    stats = annotate_tree(
        data,
        screenshots_dir=screenshots_dir,
        model_name="gpt-5.1",
        overwrite=args.overwrite,
        trajectory_path=input_path,
        transition_diff_threshold=threshold,
    )
    
    # Preserve metadata structure if it existed
    if isinstance(data, dict) and "_metadata" in data:
        output_data = {
            "_metadata": data["_metadata"],
            "segments": data.get("segments", data),
        }
        save_json(output_data, output_path)
        segments = extract_segments(output_data)
    else:
        save_json(data, output_path)
        segments = extract_segments(data)

    # Save segments to individual files (similar to processed_trajectory_segments)
    output_dir = output_base / "annotated"
    save_segments(
        segments,
        output_file=None,  # Already saved above
        output_dir=output_dir,
    )

    print(
        f"[ANNOTATE] Processed {stats['actions']} action nodes; "
        f"annotated {stats['annotated']}, skipped {stats['skipped']}."
    )
    print(f"[OUTPUT] Wrote annotations to {output_path}")
    print(f"[OUTPUT] Wrote {len(segments)} segment files to {output_dir}/")


if __name__ == "__main__":
    main()
