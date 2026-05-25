import argparse
import importlib.util
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

from language import SequenceNode, get_first_action, get_last_action, wrap_sequence, merge_nodes
from utils import call_openai, extract_segments, load_json, save_json, save_segments, call_claude

# Import _load_issue_description from 1_segment/1_annotate.py
annotate_path = ROOT / "1_segment" / "1_annotate.py"
spec = importlib.util.spec_from_file_location("annotate", annotate_path)
annotate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotate_module)
load_issue_description = annotate_module._load_issue_description

from dotenv import load_dotenv
load_dotenv()

def load_labels(labels_path: Path) -> list[str]:
    """Load activity labels from labels.txt file."""
    labels = []
    with labels_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("**") and line.startswith("-"):
                # Extract label (e.g., "- reading issue: ..." -> "reading issue")
                label_part = line.lstrip("-").strip()
                if ":" in label_part:
                    label = label_part.split(":")[0].strip()
                else:
                    label = label_part
                if label:
                    labels.append(label)
    return labels

def get_step_goals(segments: list[dict[str, Any]]) -> str:
    """Extract numbered goal lines using only node annotations."""
    goals = []
    for i, segment in enumerate(segments):
        goal_text = segment.get("annotation", None)
        if goal_text and goal_text.strip():
            prefix = f"[{i}]"
            goals.append(f"{prefix} {goal_text}")
    return "\n".join(goals)


def get_induce_prompt(labels: list[str], issue_context: str | None = None) -> str:
    """Generate the induction prompt with coding activity categories."""
    labels_text = "\n".join([f"- {label}" for label in labels])
    
    issue_section = ""
    if issue_context:
        issue_section = f"""

Issue context (the task being worked on):
{issue_context}
"""
    
    return f"""
Your task is to summarize the general workflow from the provided task-solving steps, grouping actions into coding activities.

Available coding activity categories:
{labels_text}
{issue_section}
Note: The user is working on a coding/software development task. Group actions into the coding activity categories above. 
Use the issue context to understand how each step relates to the overall task. Infer the purpose of actions based on the issue context.

Format: [start-end] (label) Description
- Use ranges like [3-5] for consecutive steps, [7-7] for singletons
- Indices must be ascending and sequential (e.g., [182-192], not [192-182])
- Output steps in ascending order by their start index (e.g., [1-2] before [5-7], not [5-7] before [1-2])
- Label each item with its primary activity in parentheses

Example:
```
[1] Creates a new empty Google Sheet named "april-attendance-data" in chrome browser.
[2] Scrolls down the google sheet to view more rows.
[3] Scrolls down and then clicks on the top row.
[4] Enters the text "Editor" into the select cell in Google Sheet.
[5] Copies the text "Editor" into the select cell in Google Sheet.
[6] Downloads the Google Sheet as a csv file.
```

Output:
```
[1-1] (setup) Create a new empty Google Sheet named "april-attendance-data" in chrome browser.
[2-3] (reading code) View the new data in the Google Sheet.
[4-5] (writing code) Edit the new data in the Google Sheet.
[6-6] (git) Download the Google Sheet as a csv file.
```

Rules:
1. Group consecutive steps with the same activity. Use issue context to infer purpose of ambiguous actions.
2. "writing code" = non-test code files. "writing tests" = test files/code. Creating test files → "writing tests", NOT "writing code". Running tests → "test running tests". Manual testing → "test manually checking".
3. Only assign AI activities to steps where AI is mentioned (Cursor, Claude, Copilot, AI-generated)
4. "unrelated" ONLY for non-development (YouTube, music, social media). Development activities are NEVER unrelated.
5. Summarize the goal, not just concatenate steps. Maintain key details (file names, functions, AI tools).
6. Split steps that mix distinct intents (reading vs writing vs testing, AI vs non-AI).
7. Do not reorder steps. Choose the most specific label considering action + issue context.
8. Consistency: When labeling a step, check if similar actions appeared earlier in the workflow. If a similar action was already labeled, use the same label unless the context clearly indicates a different purpose. 
9. "setup" ONLY for initial environment preparation that occurs at the beginning of a work session, BEFORE starting the actual task work.
"""

def normalize_workflow_brackets(step: str) -> str:
    """Normalize bracket indices like [1,3-4,8] into a single [1-8] range."""
    match = re.search(r"\[([0-9,\-\s]+)\]", step)
    if not match:
        return step

    raw = match.group(1)
    values: set[int] = set()
    for start_str, end_str in re.findall(r"(\d+)(?:-(\d+))?", raw):
        start = int(start_str)
        end = int(end_str) if end_str else start
        low, high = (start, end) if start <= end else (end, start)
        values.update(range(low, high + 1))

    if not values:
        return step

    sorted_vals = sorted(values)
    normalized = f"[{sorted_vals[0]}-{sorted_vals[-1]}]"
    return step[:match.start()] + normalized + step[match.end():]


def clamp_adjacent_overlaps(workflow_lines: list[str]) -> list[str]:
    """Order by start index and clamp overlaps globally (back-to-front)."""
    parsed_lines: list[tuple[int, int, str]] = []
    unparsable: list[str] = []

    for line in workflow_lines:
        parsed = parse_workflow_step(line)
        if not parsed:
            unparsable.append(line)
            continue
        start, end, _, _ = parsed
        parsed_lines.append((start, end, line))

    if not parsed_lines:
        return workflow_lines

    # Sort by start index first so downstream ordering check has a stable base.
    parsed_lines.sort(key=lambda x: x[0])

    clamped_reversed: list[tuple[int, int, str]] = []
    next_start: int | None = None

    # Back-to-front pass: each current range must end before next range starts.
    for start, end, line in reversed(parsed_lines):
        if next_start is not None:
            # If start is not strictly before next_start, there is no valid non-overlapping range.
            if start >= next_start:
                continue
            if end >= next_start:
                end = next_start - 1

        clamped_reversed.append((start, end, line))
        next_start = start

    clamped = list(reversed(clamped_reversed))
    rewritten = [
        re.sub(r"^\[\d+(?:-\d+)?\]", f"[{start}-{end}]", line, count=1)
        for start, end, line in clamped
    ]

    # Keep unparsable lines so validation can still surface format issues.
    return rewritten + unparsable


def validate_workflow_steps(workflow_lines: list[str]) -> list[str]:
    """
    Validate workflow steps for parsing, ordering, and overlapping ranges.
    
    Args:
        workflow_lines: List of workflow step strings like '[182-192] (writing code) Description'
        
    Returns:
        List of error messages for failed validations. Empty list if all validations pass.
    """
    failed_steps = []
    parsed_steps = []
    
    # Parse all steps
    for step in workflow_lines:
        parsed = parse_workflow_step(step)
        if not parsed:
            failed_steps.append(step)
        else:
            parsed_steps.append((step, parsed))
    
    # If parsing failed, return early
    if failed_steps:
        return failed_steps
    
    if not parsed_steps:
        return []
    
    # Check 1: Indices must be ascending within each range (start <= end)
    # This is already validated in parse_workflow_step, but double-check
    invalid_ranges = []
    for step, (start, end, _, _) in parsed_steps:
        if start > end:
            invalid_ranges.append(f"{step} has start ({start}) > end ({end})")
    
    if invalid_ranges:
        failed_steps.extend(invalid_ranges)
    
    # Check 2: Steps must be in ascending order by start index
    out_of_order = []
    for i in range(len(parsed_steps) - 1):
        _, (start1, _, _, _) = parsed_steps[i]
        step2, (start2, _, _, _) = parsed_steps[i + 1]
        if start1 > start2:
            out_of_order.append(f"{parsed_steps[i][0]} (start={start1}) comes before {step2} (start={start2})")
    
    if out_of_order:
        failed_steps.extend(out_of_order)
    
    # Check 3: No overlapping ranges
    if not failed_steps:
        # Sort by start index to check overlaps
        sorted_parsed = sorted(parsed_steps, key=lambda x: x[1][0])
        
        overlapping_pairs = []
        for i in range(len(sorted_parsed) - 1):
            _, (start1, end1, _, _) = sorted_parsed[i]
            step2, (start2, end2, _, _) = sorted_parsed[i + 1]
            
            # Check if ranges overlap: [start1-end1] and [start2-end2] overlap if
            # start1 <= end2 and start2 <= end1
            if start1 <= end2 and start2 <= end1:
                overlapping_pairs.append((sorted_parsed[i][0], step2))
        
        if overlapping_pairs:
            failed_steps.extend([f"{s1} overlaps with {s2}" for s1, s2 in overlapping_pairs])
    
    return failed_steps


def _filter_goals_from_index(goals_text: str, start_idx: int) -> str:
    """Keep only goal lines whose index is >= start_idx."""
    filtered = []
    for line in goals_text.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            idx_str = line.split("]", 1)[0].lstrip("[")
            idx = int(idx_str)
        except Exception:
            continue
        if idx >= start_idx:
            filtered.append(line)
    return "\n".join(filtered)


def get_workflow(
    text: str,
    labels: list[str],
    issue_context: str | None = None,
    verbose: bool = True,
    max_retries: int = 2,
) -> list[str]:
    """Call the induction prompt and return bracketed workflow lines, sorted by start index.
    
    Args:
        text: Goal lines (one per segment).
        labels: Activity labels.
        issue_context: Optional issue description.
        verbose: Print model output and retry messages.
        max_retries: Number of retries on validation failures.
    """
    prompt = get_induce_prompt(labels, issue_context)
    full_text = text  # keep original goals
    working_text = text
    
    for attempt in range(max_retries + 1):
        workflow = call_openai(prompt=prompt, content=working_text, model_name="gpt-5.1")
        # workflow = call_claude(prompt=prompt, content=text, model_name="claude-sonnet-4-20250514")
        workflow = workflow.strip("```").strip("\n").strip()
        if verbose:
            print(workflow)
        workflow_lines = [normalize_workflow_brackets(ws) for ws in workflow.split("\n") if ws.startswith("[")]
        workflow_lines = clamp_adjacent_overlaps(workflow_lines)
        
        # Validate workflow steps
        failed_steps = validate_workflow_steps(workflow_lines)
        
        if not failed_steps:
            # All steps parsed successfully and validated, sort and return
            def get_start_index(step: str) -> int:
                """Extract the start index from a workflow step line."""
                parsed = parse_workflow_step(step)
                if not parsed:
                    raise ValueError(f"Unexpected parsing failure for step: {step}")
                return parsed[0]  # start_idx is first element
            
            workflow_lines.sort(key=get_start_index)
            return workflow_lines
        
        # Parsing failed - retry with enhanced prompt
        if attempt < max_retries:
            if verbose:
                print(f"\n[RETRY {attempt + 1}/{max_retries}] Validation failed, retrying...")
            
            # Enhance prompt based on error types
            has_parsing = any(not ("overlaps" in s or "comes before" in s or "start" in s and "end" in s and ">" in s) for s in failed_steps)
            has_overlaps = any("overlaps with" in s for s in failed_steps)
            has_order = any("comes before" in s for s in failed_steps)
            has_range = any("start" in s and "end" in s and ">" in s for s in failed_steps)
            
            retry_guidance = []
            if has_parsing:
                retry_guidance.append("- Ensure ALL steps follow the exact format: [start-end] (label) Description")
                retry_guidance.append("- Each step must have parentheses around the label: (label)")
                retry_guidance.append("- Do not use comma-separated indices like [1,3,5], use ranges like [1-5]")
            
            if has_range:
                retry_guidance.append("- CRITICAL: start index must be <= end index (e.g., [3-5] not [5-3])")
            
            if has_order:
                retry_guidance.append("- CRITICAL: Output steps in ascending order by start index (e.g., [1-2] before [5-7])")
                retry_guidance.append("- Sort all steps by their start index before outputting")
            
            if has_overlaps:
                retry_guidance.append("- CRITICAL: NO overlapping ranges allowed")
                retry_guidance.append("- Each step must cover unique, non-overlapping index ranges")
                retry_guidance.append("- If steps share indices, merge them into a single step or split them properly")
            
            if retry_guidance:
                retry_prompt = prompt + "\n\nIMPORTANT - Previous attempt had validation errors. Please fix:\n" + "\n".join(retry_guidance)
                if failed_steps:
                    retry_prompt += f"\n\nFailed steps from previous attempt:\n" + "\n".join(failed_steps[:5])  # Show first 5 errors
                prompt = retry_prompt

            # If the only issue is ordering, trim the goals to the offending start index and retry
            if has_order and not (has_parsing or has_overlaps or has_range):
                offending = None
                max_seen = -1
                for ws in workflow_lines:
                    parsed = parse_workflow_step(ws)
                    if not parsed:
                        continue
                    start = parsed[0]
                    if start < max_seen:
                        offending = start
                        break
                    max_seen = start
                if offending is not None:
                    working_text = _filter_goals_from_index(full_text, offending)
                    if verbose:
                        print(f"[ORDER FIX] Re-prompting from step index {offending} with filtered goals.")
                else:
                    working_text = full_text
            else:
                working_text = full_text
        else:
            # Final attempt failed
            error_msg = f"Failed to parse {len(failed_steps)} workflow step(s) after {max_retries + 1} attempts.\n"
            error_msg += "Failed steps:\n" + "\n".join(failed_steps)
            raise ValueError(error_msg)
    
    # Should never reach here, but just in case
    raise ValueError("Unexpected error in get_workflow")


def parse_workflow_step(step: str) -> tuple[int, int, str, str] | None:
    """
    Parse a workflow step line like '[182-192] (writing code) Some description'.
    
    Returns:
        Tuple of (start_idx, end_idx, label, annotation) or None if parsing fails.
        Validates that start <= end.
    """
    # Match pattern: [start-end] (label) annotation
    pattern = r'\[(\d+)-(\d+)\]\s*\(([^)]+)\)\s*(.+)'
    match = re.match(pattern, step)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            return None  # Invalid: start > end
        label = match.group(3).strip()
        annotation = match.group(4).strip()
        return (start, end, label, annotation)
    
    # Try singleton pattern: [182-182] or [182] (label) annotation
    pattern_single = r'\[(\d+)(?:-(\d+))?\]\s*\(([^)]+)\)\s*(.+)'
    match = re.match(pattern_single, step)
    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if start > end:
            return None  # Invalid: start > end
        label = match.group(3).strip()
        annotation = match.group(4).strip()
        return (start, end, label, annotation)
    
    return None


def create_grouped_segments(
    workflow_steps: list[str],
    original_segments: list[dict[str, Any]],
    verbose: bool = False
) -> list[dict[str, Any]]:
    """
    Create new segments by grouping original segments based on workflow steps.
    
    Args:
        workflow_steps: List of workflow step strings like '[182-192] (writing code) Description'
        original_segments: List of original segment dictionaries
        verbose: Print progress information
        
    Returns:
        List of new grouped segment dictionaries with labels and annotations
    """
    grouped_segments = []
    
    for step in workflow_steps:
        parsed = parse_workflow_step(step)
        if not parsed:
            if verbose:
                print(f"Warning: Could not parse workflow step: {step}")
            continue
        
        start_idx, end_idx, label, annotation = parsed
        
        # Validate indices
        if start_idx < 0 or end_idx >= len(original_segments) or start_idx > end_idx:
            if verbose:
                print(f"Warning: Invalid indices [{start_idx}-{end_idx}] for {len(original_segments)} segments")
            continue
        
        # Collect segments to group
        segments_to_group = []
        for i in range(start_idx, end_idx + 1):
            segments_to_group.append(original_segments[i])
        
        if not segments_to_group:
            if verbose:
                print(f"Warning: No segments found for range [{start_idx}-{end_idx}]")
            continue
        
        # Use merge_nodes to create grouped segment structure
        grouped_segment = merge_nodes(segments_to_group)
        
        # Get time range from first and last action nodes using utility functions
        first_segment = original_segments[start_idx]
        last_segment = original_segments[end_idx]
        
        first_action = get_first_action(first_segment)
        last_action = get_last_action(last_segment)
        
        first_time = None
        last_time = None
        
        if first_action and first_action.get("time"):
            first_time = first_action.get("time", {}).get("before")
        if last_action and last_action.get("time"):
            last_time = last_action.get("time", {}).get("after")
        
        # Add metadata to grouped segment
        grouped_segment["annotation"] = annotation
        grouped_segment["label"] = label
        grouped_segment["source_indices"] = list(range(start_idx, end_idx + 1))
        
        if first_time and last_time:
            grouped_segment["time"] = {
                "start": first_time,
                "end": last_time,
            }
        
        grouped_segments.append(grouped_segment)
        
        if verbose:
            print(f"Created grouped segment [{start_idx}-{end_idx}]: {label} - {annotation[:50]}...")
    
    return grouped_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to annotated_trajectory.json (sequence root).")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels.txt file (default: labels.txt in same directory).")
    parser.add_argument("--verbose", action="store_true", help="Print details.")
    args = parser.parse_args()

    # Load labels
    if args.labels:
        labels_path = Path(args.labels)
    else:
        labels_path = Path(__file__).parent / "labels.txt"
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels = load_labels(labels_path)
    if args.verbose:
        print(f"Loaded {len(labels)} activity labels from {labels_path}")

    data_path = Path(args.data)
    data = load_json(data_path)
    segments = extract_segments(data)
    step_goals = get_step_goals(segments)
    if args.verbose:
        print("Step goals:\n", step_goals)
    if not step_goals.strip():
        raise ValueError(f"No step goals found in {len(segments)} segments.")

    # Load issue context if available
    issue_context = load_issue_description(data_path)
    if issue_context:
        if args.verbose:
            print(f"\nLoaded issue context ({len(issue_context)} chars)")
    elif args.verbose:
        print("\nNo issue context found (issue.json not available)")

    workflow_steps = get_workflow(step_goals, labels, issue_context, verbose=args.verbose)
    print("\n--- Induced Workflow (Grouped by Coding Activities) ---")
    workflow_text = "\n".join(workflow_steps)
    print(workflow_text)
    
    # Organize outputs into subdirectories
    output_base = data_path.parent / "2_induction"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save workflow to file
    workflow_path = output_base / "workflow.txt"
    with open(workflow_path, "w") as f:
        f.write(workflow_text)
    print(f"\nSaved workflow to: {workflow_path}")
    
    # Create grouped segments from workflow
    grouped_segments = create_grouped_segments(workflow_steps, segments, verbose=args.verbose)
    
    if grouped_segments:
        # Save grouped segments
        output_path = output_base / "grouped.json"
        save_json(grouped_segments, output_path)
        print(f"\nSaved {len(grouped_segments)} grouped segments to: {output_path}")
        
        # Save individual segment files
        output_dir = output_base / "grouped"
        save_segments(
            grouped_segments,
            output_file=None,  # Already saved above
            output_dir=output_dir,
        )
        print(f"Saved {len(grouped_segments)} segment files to: {output_dir}/")
    else:
        print("\nWarning: No grouped segments were created.")


if __name__ == "__main__":
    main()
