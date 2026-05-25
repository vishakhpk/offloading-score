"""Extract a flat AI timeline from SpecStory logs."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Initialize package (sets up sys.path)
_init_path = Path(__file__).parent / "__init__.py"
exec(_init_path.read_text())


@dataclass
class LogEntry:
    role: str
    timestamp: Optional[float]
    content: str
    tool_name: Optional[str] = None
    model: Optional[str] = None
    is_sidechain: bool = False


def parse_timestamp(raw: str) -> Optional[float]:
    """Convert a variety of timestamp formats to epoch seconds."""
    if not raw:
        return None

    ts = raw.split("•")[0].strip()
    if not ts:
        return None

    formats = [
        "%Y-%m-%d %H:%MZ",
        "%Y-%m-%d %H:%M:%SZ",
        "%Y-%m-%d_%H-%M-%SZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def clean_session_title(title: str) -> str:
    title = title.strip()
    if not title:
        return title
    if "(" in title and title.endswith(")"):
        return title.rsplit("(", 1)[0].strip()
    return title


def parse_log(log_path: str) -> Tuple[List[LogEntry], dict]:
    """Parse a SpecStory markdown log into ordered log entries."""
    entries: List[LogEntry] = []
    metadata = {
        "session_id": None,
        "session_title": None,
        "session_timestamp": None,
        "interface": None,
    }

    with open(log_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Discover metadata from the header
    for line in lines:
        if metadata["session_id"] is None:
            m = re.search(
                r"<!--\s*(cursor|Claude Code) Session ([a-f0-9\-]+)(?:\s*\(([^)]+)\))?",
                line,
                re.IGNORECASE,
            )
            if m:
                tool_name = m.group(1).lower()
                metadata["interface"] = "claude_code" if "claude" in tool_name else "cursor"
                metadata["session_id"] = m.group(2)
                if m.group(3) and metadata["session_timestamp"] is None:
                    ts = parse_timestamp(m.group(3))
                    if ts:
                        metadata["session_timestamp"] = ts

        if metadata["session_title"] is None:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                metadata["session_title"] = clean_session_title(title)
                try:
                    dt = datetime.strptime(title, "%Y-%m-%d %H:%M:%SZ")
                    metadata["session_timestamp"] = dt.replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    ts_match = re.search(
                        r"\((\d{4}-\d{2}-\d{2}[\s_]\d{2}[-:]\d{2}[-:]?\d{0,2}Z?)\)\s*$",
                        title,
                    )
                    if ts_match and metadata["session_timestamp"] is None:
                        ts = parse_timestamp(ts_match.group(1))
                        if ts:
                            metadata["session_timestamp"] = ts

        if metadata["session_id"] and metadata["session_title"]:
            break

    if metadata["session_timestamp"] is None:
        filename = os.path.basename(log_path)
        fn_match = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}Z)", filename)
        if fn_match:
            ts = parse_timestamp(fn_match.group(1))
            if ts:
                metadata["session_timestamp"] = ts

    i = 0
    n = len(lines)

    entry_pattern = re.compile(
        r"_\*\*(User|Assistant|Agent)"
        r"(?:\s+\(([^)]+)\))?"
        r"(?:\s+\(([^)]+)\))?"
        r"(?:\s+\(([^)]+)\))?"
        r"\*\*_"
    )
    tool_pattern = re.compile(r"Tool use:\s+\*\*(.+?)\*\*")
    tool_use_pattern = re.compile(r"<tool-use\s+data-tool-type=\"([^\"]+)\"\s+data-tool-name=\"([^\"]+)\"")

    last_timestamp: Optional[float] = metadata.get("session_timestamp")
    last_role: Optional[str] = None
    last_model: Optional[str] = None

    while i < n:
        line = lines[i]
        stripped = line.strip()
        match = entry_pattern.match(stripped)
        tool_match = tool_pattern.match(stripped)
        tool_use_match = tool_use_pattern.match(stripped)

        if match:
            role_raw = match.group(1).lower()
            parens = [match.group(idx) for idx in range(2, 5) if match.group(idx)]

            timestamp = None
            model = None
            is_sidechain = False

            for paren in parens:
                paren = paren.strip()
                if " • " in paren:
                    ts_part, model_part = paren.split(" • ", 1)
                    ts = parse_timestamp(ts_part.strip())
                    if ts is not None:
                        timestamp = ts
                    model = model_part.strip()
                elif paren.lower() == "sidechain":
                    is_sidechain = True
                else:
                    ts = parse_timestamp(paren)
                    if ts is not None:
                        timestamp = ts
                    else:
                        model = paren

            last_timestamp = timestamp or last_timestamp
            last_role = role_raw
            if model:
                last_model = model
            i += 1

            while i < n and lines[i].strip() == "":
                i += 1

            content_lines: List[str] = []
            while i < n:
                current = lines[i]
                if current.strip() == "---":
                    break
                if entry_pattern.match(current.strip()):
                    break
                content_lines.append(current.rstrip("\n"))
                i += 1

            while content_lines and not content_lines[-1].strip():
                content_lines.pop()

            content = "\n".join(content_lines).strip()
            if content:
                entries.append(
                    LogEntry(
                        role=role_raw,
                        timestamp=timestamp,
                        content=content,
                        model=model,
                        is_sidechain=is_sidechain,
                    )
                )

            while i < n and lines[i].strip() == "---":
                i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            continue

        if tool_match:
            tool_name = tool_match.group(1).strip()
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1

            content_lines: List[str] = []
            while i < n:
                current = lines[i]
                if current.strip() == "---":
                    break
                content_lines.append(current.rstrip("\n"))
                i += 1

            while content_lines and not content_lines[-1].strip():
                content_lines.pop()

            content = "\n".join(content_lines).strip()
            entries.append(
                LogEntry(
                    role="tool",
                    timestamp=last_timestamp,
                    content=content,
                    tool_name=tool_name,
                )
            )

            while i < n and lines[i].strip() == "---":
                i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            continue

        if tool_use_match:
            tool_name = tool_use_match.group(2)

            content_lines: List[str] = []
            while i < n:
                current = lines[i]
                content_lines.append(current.rstrip("\n"))
                if "</tool-use>" in current:
                    i += 1
                    break
                if entry_pattern.match(current.strip()):
                    break
                if current.strip() == "---":
                    break
                i += 1

            content = "\n".join(content_lines).strip()
            entries.append(
                LogEntry(
                    role="tool",
                    timestamp=last_timestamp,
                    content=content,
                    tool_name=tool_name,
                    model=last_model,
                )
            )

            while i < n and lines[i].strip() == "---":
                i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            continue

        if stripped and stripped != "---" and last_role in ("assistant", "agent"):
            content_lines: List[str] = []
            while i < n:
                current = lines[i]
                current_stripped = current.strip()
                if current_stripped == "---":
                    break
                if entry_pattern.match(current_stripped):
                    break
                if tool_pattern.match(current_stripped):
                    break
                if tool_use_pattern.match(current_stripped):
                    break
                content_lines.append(current.rstrip("\n"))
                i += 1

            while content_lines and not content_lines[-1].strip():
                content_lines.pop()

            content = "\n".join(content_lines).strip()
            if content:
                entries.append(
                    LogEntry(
                        role=last_role,
                        timestamp=last_timestamp,
                        content=content,
                        model=last_model,
                    )
                )

            while i < n and lines[i].strip() == "---":
                i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            continue

        i += 1

    return entries, metadata


def extract_summary_from_content(content: str) -> Optional[str]:
    match = re.search(r"<summary>([^<]+)</summary>", content)
    if match:
        return match.group(1).strip()
    return None


def extract_tool_name_from_content(content: str) -> Optional[str]:
    match = re.search(r'<tool-use[^>]*data-tool-name="([^"]+)"', content)
    if match:
        return match.group(1)

    match = re.search(r"Tool use:\s+\*\*(.+?)\*\*", content)
    if match:
        return match.group(1)

    return None


def classify_ai_event_type(content: str, role: str, tool_name: Optional[str]) -> str:
    if role == "user":
        return "user_request"

    if tool_name is not None:
        return "tool_call"
    if content.strip().startswith("Tool use:"):
        return "tool_call"
    if content.strip().startswith("<tool-use"):
        return "tool_call"

    if "<think>" in content:
        return "ai_thinking"

    return "ai_response"


def gather_log_paths(log_path: str) -> List[str]:
    resolved = os.path.abspath(log_path)

    if os.path.isfile(resolved):
        if not resolved.endswith(".md"):
            raise ValueError("Expected a markdown log file.")
        return [resolved]

    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Log path not found: {resolved}")

    candidate_dirs = []
    if resolved.endswith(".specstory"):
        candidate_dirs.append(os.path.join(resolved, "history"))
    if os.path.basename(resolved) == "history":
        candidate_dirs.append(resolved)
    candidate_dirs.append(os.path.join(resolved, "history"))
    candidate_dirs.append(resolved)

    history_dir = None
    for candidate in candidate_dirs:
        if os.path.isdir(candidate):
            history_dir = candidate
            break

    if history_dir is None:
        raise FileNotFoundError(f"No history directory found for: {resolved}")

    log_paths = sorted(
        str(path)
        for path in Path(history_dir).glob("*.md")
        if path.is_file()
    )

    if not log_paths:
        raise FileNotFoundError(f"No markdown logs found in: {history_dir}")

    return log_paths


def extract_ai_timeline(log_path: str, output_path: Optional[str], indent: int = 2) -> dict:
    log_paths = gather_log_paths(log_path)

    ai_timeline: List[dict] = []
    exchanges: List[dict] = []
    sessions: List[dict] = []

    exchange_index = 0

    for log_file in log_paths:
        entries, metadata = parse_log(log_file)
        session_id = metadata.get("session_id")
        session_title = metadata.get("session_title")
        session_tool = metadata.get("interface", "cursor")
        session_file = os.path.basename(log_file)

        if entries:
            first_ts = next((e.timestamp for e in entries if e.timestamp), None)
            last_ts = next((e.timestamp for e in reversed(entries) if e.timestamp), None)
            sessions.append(
                {
                    "id": session_id,
                    "title": session_title,
                    "interface": session_tool,
                    "file": session_file,
                    "time": [first_ts, last_ts],
                }
            )

        current_exchange_events: List[dict] = []
        current_user_request = None
        last_timestamp = metadata.get("session_timestamp")

        for entry in entries:
            timestamp = entry.timestamp or last_timestamp
            if timestamp is None:
                continue
            if entry.timestamp:
                last_timestamp = entry.timestamp

            event_type = classify_ai_event_type(entry.content, entry.role, entry.tool_name)
            summary = extract_summary_from_content(entry.content)

            event = {
                "timestamp": timestamp,
                "type": event_type,
                "exchange_index": exchange_index,
                "interface": session_tool,
                "content": entry.content,
            }

            if entry.role != "user":
                event["role"] = entry.role
            if entry.model:
                event["model"] = entry.model
            tool_name = entry.tool_name
            if not tool_name and event_type == "tool_call":
                tool_name = extract_tool_name_from_content(entry.content)
            if tool_name:
                event["tool_name"] = tool_name
            if summary:
                event["summary"] = summary
            if entry.is_sidechain:
                event["is_sidechain"] = True

            ai_timeline.append(event)
            current_exchange_events.append(event)

            if event_type == "user_request":
                if current_user_request is not None:
                    exchanges.append(
                        {
                            "index": exchange_index - 1,
                            "user_request": current_user_request,
                            "event_count": len(current_exchange_events) - 1,
                            "session_file": session_file,
                        }
                    )
                current_user_request = {
                    "timestamp": timestamp,
                    "content": entry.content,
                }
                current_exchange_events = [event]
                exchange_index += 1

        if current_user_request is not None:
            exchanges.append(
                {
                    "index": exchange_index - 1,
                    "user_request": current_user_request,
                    "event_count": len(current_exchange_events),
                    "session_file": session_file,
                }
            )

    # Sort AI timeline chronologically (oldest to newest by timestamp)
    # All events should have valid timestamps (None values are filtered out above)
    ai_timeline.sort(key=lambda x: x["timestamp"])
    
    # Sort exchanges chronologically by user_request timestamp
    exchanges.sort(key=lambda x: x["user_request"]["timestamp"])

    timestamps = [event["timestamp"] for event in ai_timeline]
    time_range = [min(timestamps), max(timestamps)] if timestamps else [None, None]

    event_counts = {}
    for event in ai_timeline:
        event_counts[event["type"]] = event_counts.get(event["type"], 0) + 1

    result = {
        "ai_timeline": ai_timeline,
        "exchanges": exchanges,
        "sessions": sessions,
        "time_range": time_range,
        "event_counts": event_counts,
    }

    if output_path:
        from pathlib import Path
        from utils import save_json
        
        output_file = Path(output_path)
        # Handle custom indent (save_json defaults to 2, but we allow None/0)
        if indent is None or indent < 0:
            # Use custom formatting for no indent
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as fw:
                json.dump(result, fw, indent=None)
        else:
            save_json(result, output_file)
        print(f"Saved AI timeline to: {output_path}")

    print(f"Extracted {len(ai_timeline)} AI events from {len(log_paths)} log files.")
    print(f"Event types: {event_counts}")
    print(f"Time range: {time_range[0]} - {time_range[1]}")

    return result


def _collect_action_nodes(node: dict) -> List[dict]:
    """Flatten action nodes from a trajectory dict."""
    if not isinstance(node, dict):
        return []
    if node.get("node_type") == "action":
        return [node]
    nodes = []
    for child in node.get("nodes", []):
        nodes.extend(_collect_action_nodes(child))
    return nodes


def annotate_nodes_with_ai(
    trajectory_path: str, ai_events: List[dict], output_path: Optional[str] = None
) -> None:
    """Annotate action nodes with AI events that fall within their time window."""
    from pathlib import Path
    from utils import load_json, save_json
    
    traj = load_json(Path(trajectory_path))

    action_nodes = _collect_action_nodes(traj)
    relevant_events = [
        e for e in ai_events if e.get("type") in {"user_request", "ai_response"}
    ]

    for node in action_nodes:
        time_info = node.get("time") or {}
        before = time_info.get("before")
        after = time_info.get("after")
        if not isinstance(before, (int, float)) and not isinstance(after, (int, float)):
            continue
        start = min(v for v in (before, after) if isinstance(v, (int, float)))
        end = max(v for v in (before, after) if isinstance(v, (int, float)))
        matched = []
        for e in relevant_events:
            ts = e.get("timestamp")
            if ts is None or not (start <= ts <= end):
                continue
            cleaned = {
                k: v
                for k, v in e.items()
                if k not in {"role", "model", "exchange_index"}
            }
            matched.append(cleaned)
        if matched:
            node["ai_events"] = matched

    out_path = Path(output_path or trajectory_path)
    save_json(traj, out_path)
    print(f"Annotated trajectory written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a flat AI timeline from SpecStory logs.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a SpecStory markdown log, history directory, or .specstory folder.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Destination for ai_interactions.json. Defaults next to the provided log path.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for the emitted JSON (use negative for compact).",
    )
    parser.add_argument(
        "--trajectory",
        help="Path to processed_trajectory.json to annotate with AI events.",
    )
    parser.add_argument(
        "--trajectory_out",
        help="Optional output path for annotated trajectory (defaults to in-place).",
    )

    args = parser.parse_args()

    resolved_data_dir = os.path.abspath(args.data)
    if not os.path.exists(resolved_data_dir):
        raise FileNotFoundError(f"Log path not found: {resolved_data_dir}")

    indent = None if args.indent is None or args.indent < 0 else args.indent

    if args.output_path:
        output_path = os.path.abspath(args.output_path)
    else:
        # Organize into subdirectories
        if os.path.isdir(resolved_data_dir):
            base_dir = resolved_data_dir
        else:
            base_dir = os.path.dirname(resolved_data_dir)
        output_base = os.path.join(base_dir, "0_preprocessing")
        os.makedirs(output_base, exist_ok=True)
        output_path = os.path.join(output_base, "ai_interactions.json")

    result = extract_ai_timeline(resolved_data_dir, output_path, indent=indent if indent is not None else None)

    if args.trajectory:
        traj_path = os.path.abspath(args.trajectory)
        traj_out = os.path.abspath(args.trajectory_out) if args.trajectory_out else traj_path
        annotate_nodes_with_ai(traj_path, result["ai_timeline"], output_path=traj_out)


if __name__ == "__main__":
    main()
