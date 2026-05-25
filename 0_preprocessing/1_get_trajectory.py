"""Load the process human trajectory from the database (local only)."""
from dotenv import load_dotenv

load_dotenv()

import os
import re
import shutil
import argparse
from pathlib import Path

# Initialize package (sets up sys.path)
_init_path = Path(__file__).parent / "__init__.py"
exec(_init_path.read_text())

import pandas as pd
from sqlalchemy import create_engine, text
from utils import (
    is_click_action, is_keyboard_action, is_scroll_action,
    get_key_input, compose_key_input
)
from language import ActionNode, SequenceNode

# %% Action Processing

def load_actions_from_db(
    log_dir: str,
    db_path: str,
    table_name: str = "observations",
) -> tuple[list[str], list[str], list[float | None]]:
    """Load the actions and their timestamps from the database."""
    db_path = os.path.expanduser(os.path.join(log_dir, db_path))
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as connection:
        try:
            query = text(f"SELECT * from {table_name}")
            df = pd.read_sql_query(query, connection)
            print(f"Loaded actions from '{table_name}' table.")
        except Exception as e:
            print(f"WARNING: '{table_name}' table not found ({e}). Returning empty actions.")
            return [], [], []
    actions = df["content"].to_list()
    timestamp_strings = df["created_at"].astype(str).to_list()
    created_at = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    timestamp_seconds = [
        ts.timestamp() if not pd.isna(ts) else None
        for ts in created_at
    ]
    return actions, timestamp_strings, timestamp_seconds


def parse_screenshot_filename(filename: str) -> dict | None:
    """Parse a screenshot filename into its components.
    
    Expected formats:
    - Click/scroll: "1760054790.80566_click_left(100,200)_before.jpg"
    - Keyboard: "1760054790.80566_key_press(Key.cmd)_first.jpg"
    
    Returns dict with keys: timestamp, action, suffix, path
    """
    base = os.path.basename(filename)
    if not base.endswith('.jpg'):
        return None
    
    # Extract timestamp (first part before _)
    parts = base.split('_', 1)
    if len(parts) < 2:
        return None
    
    timestamp_str = parts[0]
    try:
        timestamp = float(timestamp_str)
    except (ValueError, TypeError):
        return None
    
    remainder = parts[1]  # e.g., "click_left(100,200)_before.jpg" or "key_press(Key.cmd)_first.jpg"
    
    # Remove .jpg extension
    remainder = remainder[:-4]
    
    # Extract suffix (last part after _)
    suffix_parts = remainder.rsplit('_', 1)
    if len(suffix_parts) == 2:
        action = suffix_parts[0]
        suffix = suffix_parts[1]  # "before", "after", "first", "final"
    else:
        action = remainder
        suffix = ""
    
    event_id = f"{timestamp_str}_{action}"

    return {
        "timestamp": timestamp,
        "timestamp_str": timestamp_str,
        "action": action,
        "suffix": suffix,
        "filename": filename,
        "event_id": event_id,
    }


def load_actions_from_screenshots(screenshot_dir: str) -> list[dict]:
    """Load actions from screenshot filenames as the primary source of truth.
    
    Returns a list of action dicts sorted by timestamp, each containing:
    - action: the action string
    - timestamp: float timestamp from filename
    - before_path: path to before/first screenshot
    - after_path: path to after/final screenshot
    """
    if not os.path.isdir(screenshot_dir):
        raise FileNotFoundError(f"Screenshot directory not found: {screenshot_dir}")
    
    # Parse all screenshot filenames
    screenshots = []
    for filename in os.listdir(screenshot_dir):
        parsed = parse_screenshot_filename(filename)
        if parsed:
            parsed["path"] = os.path.join(screenshot_dir, filename)
            screenshots.append(parsed)
    
    # Sort by timestamp
    screenshots.sort(key=lambda x: x["timestamp"])
    
    # Group screenshots by action instance (timestamp + action string)
    # For each action, we expect a before/first and after/final pair
    action_groups = {}  # key: event_id -> list of screenshots
    
    for ss in screenshots:
        key = ss.get("event_id")
        if key is None:
            timestamp_label = ss.get("timestamp_str")
            if timestamp_label is None:
                timestamp_label = f"{ss['timestamp']:.6f}"
            key = f"{timestamp_label}_{ss['action']}"
            ss["event_id"] = key
        if key not in action_groups:
            action_groups[key] = []
        action_groups[key].append(ss)
    
    # Build action list from grouped screenshots
    actions = []
    processed_keys = set()
    
    for ss in screenshots:
        key = ss.get("event_id")
        if key in processed_keys:
            continue
        processed_keys.add(key)
        
        group = action_groups[key]
        
        # Find before/first and after/final screenshots
        before_path = None
        after_path = None
        min_ts = min(s["timestamp"] for s in group)
        max_ts = max(s["timestamp"] for s in group)
        
        for s in group:
            if s["suffix"] in ("before", "first"):
                before_path = s["path"]
            elif s["suffix"] in ("after", "final"):
                after_path = s["path"]
        
        # If we only have one screenshot, use it for both
        if before_path is None and after_path is not None:
            before_path = after_path
        elif after_path is None and before_path is not None:
            after_path = before_path
        
        if before_path is None and after_path is None:
            # Use the first screenshot in the group
            before_path = group[0]["path"]
            after_path = group[-1]["path"] if len(group) > 1 else before_path
        
        actions.append({
            "action": ss["action"],
            "timestamp": min_ts,
            "timestamp_end": max_ts,
            "before_path": before_path,
            "after_path": after_path,
            "has_screenshot": True,
        })
    
    # Sort by timestamp
    actions.sort(key=lambda x: x["timestamp"])
    
    print(f"Loaded {len(actions)} actions from screenshots.")
    return actions


def merge_db_keypresses_into_screenshot_actions(
    screenshot_actions: list[dict],
    db_actions: list[str],
    db_timestamps: list[float | None],
) -> list[dict]:
    """Merge additional keypresses from DB that don't have corresponding screenshots.
    
    For each DB keypress action, check if there's a matching screenshot action
    (same action, close timestamp). If not, insert it into the timeline.
    """
    # Find DB keypresses without screenshots
    orphan_keypresses = []
    for action, ts in zip(db_actions, db_timestamps):
        if not is_keyboard_action(action):
            continue
        if ts is None:
            continue
        
        # Normalize action for comparison
        norm_action = action.replace("'", "").strip()
        
        # Check if this keypress has a matching screenshot
        has_match = False
        for sa in screenshot_actions:
            sa_norm = sa["action"].replace("'", "").strip()
            if sa_norm == norm_action and abs(sa["timestamp"] - ts) < 2.0:
                has_match = True
                break
        
        if not has_match:
            orphan_keypresses.append({
                "action": action,
                "timestamp": ts,
                "timestamp_end": ts,
                "before_path": None,
                "after_path": None,
                "has_screenshot": False,
            })
    
    print(f"Found {len(orphan_keypresses)} keypresses from DB without screenshots.")
    
    # Merge orphan keypresses into screenshot actions
    all_actions = screenshot_actions + orphan_keypresses
    all_actions.sort(key=lambda x: x["timestamp"])
    
    # First merge and sort
    all_actions = screenshot_actions + orphan_keypresses
    all_actions.sort(key=lambda x: x["timestamp"])

    # Combine consecutive scroll actions
    merged_scroll: list[dict] = []
    scroll_buffer: list[dict] = []

    def flush_scroll_buffer():
        nonlocal scroll_buffer, merged_scroll
        if not scroll_buffer:
            return
        combined = combine_scroll_actions(scroll_buffer)
        merged_scroll.append(combined)
        scroll_buffer = []

    for entry in all_actions:
        if is_scroll_action(entry["action"]):
            scroll_buffer.append(entry)
            continue
        flush_scroll_buffer()
        merged_scroll.append(entry)
    flush_scroll_buffer()

    # Combine consecutive keyboard actions when they are simple text (no Key. or modifiers)
    merged_keyboard: list[dict] = []
    kb_buffer: list[dict] = []

    def flush_kb_buffer():
        nonlocal kb_buffer, merged_keyboard
        if not kb_buffer:
            return
        keys = [get_key_input(e["action"]) for e in kb_buffer]
        simple = all(k and "key." not in k.lower() and "+" not in k and len(k) == 1 for k in keys)
        if len(kb_buffer) == 1 or not simple:
            for entry in kb_buffer:
                entry = dict(entry)
                entry.pop("has_screenshot", None)
                merged_keyboard.append(entry)
        else:
            combined = compose_key_input(keys)
            action_str = f"key_press('{combined}')" if combined else kb_buffer[0]["action"]
            merged_keyboard.append(
                {
                    "action": action_str,
                    "timestamp": kb_buffer[0]["timestamp"],
                    "timestamp_end": kb_buffer[-1].get("timestamp_end", kb_buffer[-1]["timestamp"]),
                    "before_path": kb_buffer[0].get("before_path"),
                    "after_path": kb_buffer[-1].get("after_path"),
                    "merged_from": [e["action"] for e in kb_buffer],
                }
            )
        kb_buffer = []

    for entry in merged_scroll:
        if is_keyboard_action(entry["action"]):
            kb_buffer.append(entry)
            continue
        flush_kb_buffer()
        entry = dict(entry)
        entry.pop("has_screenshot", None)
        merged_keyboard.append(entry)
    flush_kb_buffer()

    for e in merged_keyboard:
        e.pop("has_screenshot", None)

    merged_keyboard.sort(key=lambda x: x["timestamp"])
    return merged_keyboard

def _normalize_action(action: str) -> str:
    return action.lower().replace("'", "").strip()


def hotkey_in_action(action: str) -> bool:
    """Check if the action contains a hotkey (expanded combos)."""
    normalized = _normalize_action(action)
    return any(hk in normalized for hk in HOTKEY_COMBOS)


def is_modifier_key(action: str) -> bool:
    """Check if the action is a modifier key press (cmd, ctrl, alt, shift)."""
    if not is_keyboard_action(action):
        return False
    action_lower = action.lower()
    modifiers = ['.cmd', '.ctrl', '.alt', '.shift', '.meta', 'key.cmd', 'key.ctrl', 'key.alt', 'key.shift']
    return any(mod in action_lower for mod in modifiers)


def get_modifier_name(action: str) -> str:
    """Extract the modifier name from a key press action."""
    action_lower = action.lower()
    if '.cmd' in action_lower or 'key.cmd' in action_lower or '.meta' in action_lower or 'key.meta' in action_lower:
        return 'cmd'
    if '.ctrl' in action_lower or 'key.ctrl' in action_lower:
        return 'ctrl'
    if '.alt' in action_lower or 'key.alt' in action_lower:
        return 'alt'
    if '.shift' in action_lower or 'key.shift' in action_lower:
        return 'shift'
    return ''


_BASE_HOTKEY_SUFFIXES = {
    "c", "v", "x", "z", "s", "a", "enter", "k", "p", "/", "f", "g", "d", "l", "t", "w", "n", "q", "o"
    "shift+z", "shift+/", "shift+p", "shift+n", "shift+t", "shift+w",
    "shift+[", "shift+]", "shift+\\", "shift+{", "shift+}",
}

HOTKEY_COMBOS: set[str] = set()
for mod in ("ctrl", "cmd"):
    for suffix in _BASE_HOTKEY_SUFFIXES:
        HOTKEY_COMBOS.add(f"{mod}+{suffix}")
        if not suffix.startswith("shift+"):
            HOTKEY_COMBOS.add(f"{mod}+shift+{suffix}")
HOTKEY_COMBOS.update(
    {
        "alt+tab",
        "ctrl+tab",
        "ctrl+shift+tab",
        "cmd+tab",
        "cmd+shift+tab",
    }
)


def _normalize_hotkey_key_value(key_value: str) -> str:
    """Normalize key inputs for hotkey comparison (e.g., Key.enter+ -> enter)."""
    if not key_value:
        return ""
    normalized = key_value.lower().rstrip('+')
    if normalized.startswith("key."):
        normalized = normalized[4:]
    return normalized.strip()


def combine_modifier_key_combos(action_list: list[dict]) -> list[dict]:
    """Combine adjacent modifier+key presses when they match known hotkey combos."""
    if not action_list:
        return action_list
    
    result = []
    skip_indices = set()  # Track indices to skip (already combined)
    
    n = len(action_list)
    
    for i in range(n):
        if i in skip_indices:
            continue
            
        current = action_list[i]
        
        # Check if current action is a modifier key
        if is_modifier_key(current["action"]):
            modifier_name = get_modifier_name(current["action"])
            if not modifier_name:
                result.append(current)
                continue

            next_index = i + 1
            if next_index >= n or next_index in skip_indices:
                result.append(current)
                continue

            next_action = action_list[next_index]
            if not (is_keyboard_action(next_action["action"]) and not is_modifier_key(next_action["action"])):
                result.append(current)
                continue

            key_value = get_key_input(next_action["action"])
            normalized_key = _normalize_hotkey_key_value(key_value)
            if not normalized_key:
                result.append(current)
                continue

            # Don't combine if the next key is already a combo (contains '+')
            if '+' in normalized_key:
                result.append(current)
                continue

            combined_key = f"{modifier_name.lower()}+{normalized_key}"
            
            # Only combine if the combo exists in HOTKEY_COMBOS
            if combined_key not in HOTKEY_COMBOS:
                result.append(current)
                continue
            
            combined_action = f"key_press('{combined_key}')"
            combined_dict = {
                "action": combined_action,
                "timestamp": current["timestamp"],
                "timestamp_end": next_action.get("timestamp_end", next_action["timestamp"]),
                "before_path": current.get("before_path"),
                "after_path": next_action.get("after_path"),
            }
            result.append(combined_dict)
            skip_indices.add(next_index)
        else:
            result.append(current)
    
    # Re-sort by timestamp to maintain chronological order
    result.sort(key=lambda x: x["timestamp"])
    
    print(f"Modifier combos: {len(action_list)} -> {len(result)} actions")
    return result


def combine_scroll_actions(scroll_actions: list[dict]) -> dict:
    """Combine multiple scroll actions into a single scroll action.
    
    Args:
        scroll_actions: List of action dicts with 'action' key containing scroll strings
    
    Returns:
        Combined action dict with merged scroll action
    """
    if not scroll_actions:
        raise ValueError("Cannot combine empty scroll actions list")
    
    if len(scroll_actions) == 1:
        return scroll_actions[0]
    
    # Join scroll action strings in order without altering contents
    combined_action = " + ".join(a["action"] for a in scroll_actions)
    
    # Create combined action dict
    first_action = scroll_actions[0]
    last_action = scroll_actions[-1]
    combined_dict = {
        "action": combined_action,
        "timestamp": first_action.get("timestamp"),
        "timestamp_end": last_action.get("timestamp_end", last_action.get("timestamp")),
        "before_path": first_action.get("before_path"),
        "after_path": last_action.get("after_path")
    }
    
    return combined_dict


def trigger_close_buffer(action: str, buffer_actions: list[str], enable_hotkey: bool = True) -> bool:
    """Time to close the buffer: 
    - Current buffer is non-empty
    - Next new key/scroll action is different from the last action in the buffer.
    """
    if len(buffer_actions) == 0:
        return False
    if is_keyboard_action(buffer_actions[-1]) and (not is_keyboard_action(action)):
        return True
    if is_scroll_action(buffer_actions[-1]) and (not is_scroll_action(action)):
        return True
    if enable_hotkey and is_keyboard_action(action) and hotkey_in_action(action):
        return True
    return False


def trigger_add_buffer(action: str, buffer_actions: list[str]) -> bool:
    """Should add the new action to the buffer.
    - Is keyboard or scroll action
    - (i) buffer is empty; (ii) last action in buffer is the same type as the new action.
    """
    if not (is_keyboard_action(action) or is_scroll_action(action)):
        return False
    if len(buffer_actions) == 0:
        return True
    if is_keyboard_action(action) and is_keyboard_action(buffer_actions[-1]):
        # print(f"Event 1: {action} | {buffer_actions[-1]}")
        return True
    if is_scroll_action(action) and is_scroll_action(buffer_actions[-1]):
        return True
    return False


def merge_actions(actions: list[str],
                  action_timestamps: list[str] | None = None,
                  action_timestamp_seconds: list[float | None] | None = None,
                  enable_hotkey: bool = True) -> list[str]:
    """Merge adjacent keyboard and scrolling actions into a single action."""
    original_actions, merged_actions = [], []
    buffer_actions, buffer_indices = [], []

    def make_entry(before_action: str, after_action: str, before_idx: int, after_idx: int) -> dict:
        entry = {
            "before": before_action,
            "after": after_action,
            "before_idx": before_idx,
            "after_idx": after_idx,
        }
        if action_timestamps:
            entry["before_timestamp"] = action_timestamps[before_idx]
            entry["after_timestamp"] = action_timestamps[after_idx]
        if action_timestamp_seconds:
            entry["before_ts"] = action_timestamp_seconds[before_idx]
            entry["after_ts"] = action_timestamp_seconds[after_idx]
        return entry

    return original_actions, merged_actions


def merge_screenshot_actions(action_list: list[dict], enable_hotkey: bool = True) -> tuple[list[dict], list[str]]:
    """Merge adjacent keyboard and scrolling actions from screenshot-based action list.
    
    Takes action dicts with keys: action, timestamp, before_path, after_path, has_screenshot
    Returns: (original_action_dicts, merged_action_strings)
    """
    original_actions, merged_actions = [], []
    buffer_actions = []  # list of action dicts
    
    def flush_buffer():
        """Flush the current buffer and produce merged action(s)."""
        nonlocal buffer_actions
        if not buffer_actions:
            return

        if is_keyboard_action(buffer_actions[0]["action"]):
            def append_single(entry: dict):
                original_actions.append({
                    "action": entry["action"],
                    "timestamp": entry["timestamp"],
                    "timestamp_end": entry.get("timestamp_end", entry["timestamp"]),
                    "before_path": entry.get("before_path"),
                    "after_path": entry.get("after_path"),
                    "has_screenshot": entry.get("has_screenshot", False),
                    "merged_from": entry.get("merged_from"),
                })
                merged_actions.append(entry["action"])

            chunk: list[dict] = []

            def flush_chunk():
                if not chunk:
                    return
                buffer_values = [get_key_input(a["action"]) for a in chunk]
                keyboard_input = compose_key_input(buffer_values)
                if not keyboard_input:
                    for entry in chunk:
                        append_single(entry)
                else:
                    merged_action = f"key_press('{keyboard_input}')"
                    original_actions.append({
                        "action": merged_action,
                        "timestamp": chunk[0]["timestamp"],
                        "timestamp_end": chunk[-1].get("timestamp_end", chunk[-1]["timestamp"]),
                        "before_path": chunk[0].get("before_path"),
                        "after_path": chunk[-1].get("after_path"),
                        "has_screenshot": any(a.get("has_screenshot", False) for a in chunk),
                        "merged_from": [a["action"] for a in chunk],
                    })
                    merged_actions.append(merged_action)
                chunk.clear()

            for entry in buffer_actions:
                if is_modifier_key(entry["action"]):
                    flush_chunk()
                    append_single(entry)
                    continue

                key_value = get_key_input(entry["action"])
                normalized_value = key_value.lower() if isinstance(key_value, str) else ""
                if (
                    entry.get("merged_modifier")
                    or normalized_value.startswith(("cmd+", "ctrl+", "alt+", "shift+"))
                ):
                    flush_chunk()
                    append_single(entry)
                else:
                    chunk.append(entry)

            flush_chunk()

        elif is_scroll_action(buffer_actions[0]["action"]):
            combined = combine_scroll_actions(buffer_actions)
            original_actions.append({
                "action": combined["action"],
                "timestamp": combined["timestamp"],
                "timestamp_end": combined.get("timestamp_end", combined["timestamp"]),
                "before_path": combined.get("before_path"),
                "after_path": combined.get("after_path"),
                "has_screenshot": combined.get("has_screenshot", False),
                "merged_from": combined.get("merged_from"),
            })
            merged_actions.append(combined["action"])

        buffer_actions = []
    
    for action_dict in action_list:
        action = action_dict["action"]
        
        # Check if we need to close the current buffer
        if buffer_actions:
            close_buffer = trigger_close_buffer(action, [a["action"] for a in buffer_actions], enable_hotkey=True)
            if close_buffer:
                flush_buffer()
        
        # Check if we should add to buffer
        add_to_buffer = trigger_add_buffer(action, [a["action"] for a in buffer_actions])
        if add_to_buffer:
            buffer_actions.append(action_dict)
        else:
            # Non-bufferable action (e.g., click)
            original_actions.append({
                "action": action,
                "timestamp": action_dict["timestamp"],
                "timestamp_end": action_dict.get("timestamp_end", action_dict["timestamp"]),
                "before_path": action_dict["before_path"],
                "after_path": action_dict["after_path"],
            })
            merged_actions.append(action)
    
    # Flush any remaining buffer
    flush_buffer()
    
    return original_actions, merged_actions


# %% State

def parse_screenshot_filename_timestamp(path: str) -> float | None:
    """Extract the numeric timestamp prefix from a screenshot filename."""
    if not path:
        return None
    base = os.path.basename(path)
    ts_str = base.split('_', 1)[0]
    try:
        return float(ts_str)
    except (ValueError, TypeError):
        return None


def find_screenshot(
    screenshot_paths: list[str],
    action: str,
    suffix: str,
    target_timestamp: float | None = None,
) -> tuple[str, list[str]]:
    """Find the screenshot path for the given action and suffix.
    Return the screenshot path and the remaining screenshot paths."""
    # Require an exact action+suffix match to avoid substring mismatches
    # Example filename: "1760054790.80566_key_press(cmd)_first.jpg"
    def extract_action_part(base_name: str) -> str:
        try:
            _, remainder = base_name.split('_', 1)
        except ValueError:
            remainder = base_name
        return remainder.rsplit('_', 1)[0]

    def normalize_action(a: str | None) -> str:
        """Normalize action strings to make filename vs. DB comparisons robust."""
        if not a:
            return ""
        # Drop quotes and whitespace so "key_press('cmd')" matches "key_press(cmd)".
        return a.replace("'", "").strip()

    if is_keyboard_action(action):
        # For keyboard actions, we have both an intended timestamp (from the DB)
        # and an action string. Use *both* to pick the best screenshot:
        # 1) filter by normalized action match, then
        # 2) choose the closest timestamp to target_timestamp (if provided).
        norm_action = normalize_action(action)
        # print(f"norm_action: {norm_action}")
        # print(f"target_timestamp: {target_timestamp}")
        
        best_idx = None
        best_dt = None

        for i, sp in enumerate(screenshot_paths):
            base = os.path.basename(sp)
            candidate_action = normalize_action(extract_action_part(base))

            if candidate_action != norm_action:
                continue

            if target_timestamp is not None:
                ts = parse_screenshot_filename_timestamp(base)
                if ts is None:
                    continue
                THRESHOLD = 0.010
                # dt = abs(ts - target_timestamp)
                # if dt <= THRESHOLD and ((best_dt is None) or (dt < best_dt)):
                #     best_idx = i
                #     best_dt = dt
                if int(ts) == int(target_timestamp):
                    best_idx = i
                    break
                elif abs(target_timestamp - ts) <= THRESHOLD:
                    best_idx = i
                    break
            else:
                # No timestamp to compare; first exact action+suffix match wins.
                best_idx = i
                break

        if best_idx is not None:
            chosen = screenshot_paths[best_idx]
            remaining = screenshot_paths[: best_idx] + screenshot_paths[best_idx + 1 :]
            return chosen, remaining

    else:
        for i, sp in enumerate(screenshot_paths):
            base = os.path.basename(sp)
            if base.endswith(f"{action}{suffix}"):
                return screenshot_paths[i], screenshot_paths[: i] + screenshot_paths[i+1:]
    
    # print("no action", action)
    # print("suffix", suffix)
    return None, screenshot_paths


def get_states(actions: list[str], screenshot_dir: str, drive_folder_name: str = None, is_windows: bool = False) -> list[dict]:
    """Get before/after states from screenshots, either local dir or Google Drive folder."""
    # print(f"drive has id {drive_folder_id}")
    if drive_folder_name:
        drive_folder_id = find_folder_by_name(drive_folder_name)
        files = list_drive_files(drive_folder_id)
        screenshot_paths = sorted(files, key=lambda f: f["name"].split("_")[0])
    elif screenshot_dir:
        screenshot_paths = sorted(os.listdir(screenshot_dir), key=lambda x: x.split('_')[0])
        screenshot_paths = [os.path.join(screenshot_dir, p) for p in screenshot_paths]
    else:
        raise ValueError("Local screenshot directory must be provided when Google Drive folder is not specified.")

    states = []

    def log_resolution(label: str, path: str | None, expected_suffix: str):
        """Print details about which screenshot path is being used."""
        if not path:
            return f"  Missing {label} screenshot (expected suffix {expected_suffix})"
        base = os.path.basename(path)
        timestamp = base.split('_', 1)[0] if '_' in base else "unknown"
        actual_suffix = '_' + base.rsplit('_', 1)[-1]
        actual_suffix = actual_suffix if actual_suffix.endswith(".jpg") else f"{actual_suffix}.jpg"
        tag = actual_suffix.replace(".jpg", "").lstrip('_').upper()
        suffix_note = ""
        if actual_suffix != expected_suffix:
            suffix_note = f" (substituted {actual_suffix})"
        return f"  ↪︎ {label.capitalize()} screenshot: {path} [{tag}, timestamp={timestamp}]{suffix_note}"

    def extract_timestamp(path: str | None) -> str:
        if not path:
            return "None"
        base = os.path.basename(path)
        return base.split('_', 1)[0] if '_' in base else "unknown"

    for idx, action_dict in enumerate(actions):
        # print('action', action_dict)
        suffix_before = "_first.jpg" if is_keyboard_action(action_dict["before"]) else "_before.jpg"
        suffix_after = "_final.jpg" if is_keyboard_action(action_dict["after"]) else "_after.jpg"

        if drive_folder_name:
            file_names = [f["name"] for f in screenshot_paths]
            # print(file_names)
            before_name, file_names = find_screenshot(
                file_names,
                action_dict["before"],
                suffix_before,
                target_timestamp=action_dict.get("before_ts"),
            )
            after_name, file_names = find_screenshot(
                file_names,
                action_dict["after"],
                suffix_after,
                target_timestamp=action_dict.get("after_ts"),
            )

            state = {"before": before_name, "after": after_name}
        else:
            before_path, screenshot_paths = find_screenshot(
                screenshot_paths,
                action_dict["before"],
                suffix_before,
                target_timestamp=action_dict.get("before_ts"),
            )
            after_path, screenshot_paths = find_screenshot(
                screenshot_paths,
                action_dict["after"],
                suffix_after,
                target_timestamp=action_dict.get("after_ts"),
            )
            state = {"before": before_path, "after": after_path}
        
        # Check for mismatches and handle interrupted keyboard sessions
        b = state["before"]
        a = state["after"]
        needs_log = b is None or a is None
        if needs_log:
            print("=" * 72)
            print(f"Action #{idx}")
            print(f"  before action: {action_dict['before']}")
            print(f"  after action : {action_dict['after']}")
            print(f"  before timestamp (db): {action_dict.get('before_timestamp', 'unknown')}")
            print(f"  after timestamp  (db): {action_dict.get('after_timestamp', 'unknown')}")
            print(f"  before screenshot timestamp: {extract_timestamp(b)}")
            print(f"  after screenshot  timestamp: {extract_timestamp(a)}")
            print(log_resolution("before", b, suffix_before))
            print(log_resolution("after", a, suffix_after))

        if b is None:
            print(f"expected_before_suffix: {suffix_before}")
            print(f"expected_after_suffix: {suffix_after}")
            print(f"  NO MATCH - Before: {b}, After: {a}")
        elif a is None:
            if is_keyboard_action(action_dict["before"]):
                print("  ↪︎ Keyboard 'after' missing (likely interrupted by next action). Will link to next 'before' during timing.")
            else:
                print(f"expected_after_suffix: {suffix_after}")
                print(f"  NO MATCH - Before: {b}, After: {a}")

        if needs_log:
            print("---- end of screenshot resolution ----")
            print("=" * 72)

        states.append(state)

    print(f"First state: {states[0] if states else 'None'}")
    return states


def adjust_states(actions: list[str], states: list[dict]) -> list[dict]:
    """Adjust the states to reflect more accurate changes."""
    adjusted_states = []
    for i, (action, state) in enumerate(zip(actions, states)):
        if (i == 0) or is_keyboard_action(action):
            before_state = state["before"]
        else:
            before_state = states[i-1]["after"]

        if is_keyboard_action(action) and (i < len(actions) - 1):
            # Only use next action's "before" screenshot if keyboard action has no valid "_final.jpg"
            # This respects the keyboard timeout mechanism that saves proper "_final.jpg" screenshots
            if state.get("after") is None:
                after_state = states[i+1]["before"]
            else:
                after_state = state["after"]
        else:
            after_state = state.get("after", state["before"])

        adjusted_states.append({"before": before_state, "after": after_state})

    return adjusted_states


# %% Merge Click Actions

def parse_click_coords(action: str) -> tuple[float, float] | None:
    """Parse click coordinates from an action string, if present."""
    if not action:
        return None
    match = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", action)
    if not match:
        return None
    try:
        return float(match.group(1)), float(match.group(2))
    except (TypeError, ValueError):
        return None

def is_double_click(step_1: ActionNode, step_2: ActionNode, time_threshold: float = 0.5, distance_threshold: float = 10) -> bool:
    """Check if the two click actions constitute a double click."""
    if not (is_click_action(step_1.action) and is_click_action(step_2.action)):
        return False
    if not step_2.time:
        return False
    time_gap = step_2.time.range if getattr(step_2, "time", None) else None
    if time_gap is None:
        return False
    if time_gap > time_threshold:
        return False

    coords_1 = parse_click_coords(step_1.action)
    coords_2 = parse_click_coords(step_2.action)
    if coords_1 is None or coords_2 is None:
        return False
    x1, y1 = coords_1
    x2, y2 = coords_2
    dx, dy = x2 - x1, y2 - y1
    distance = (dx * dx + dy * dy) ** 0.5
    return distance < distance_threshold

def merge_double_clicks(node_list: list[ActionNode]) -> list[ActionNode]:
    """Merge double clicks into a single click."""
    merged_node_list = []
    i, N = 0, len(node_list) - 1
    while i < N:
        step, next_step = node_list[i], node_list[i+1]
        if is_double_click(step, next_step):
            coords_str = '(' + step.action.split('(')[1]
            merged_action = "double_click" + coords_str

            data = {
                "action": merged_action,
                "state": {
                    "before": step.state.before,
                    "after": next_step.state.after,
                },
                "time": {
                    "before": step.time.before,
                    "after": next_step.time.after,
                    "range": (step.time.range or 0.0) + (next_step.time.range or 0.0),
                }
            }
            merged_node_list.append(ActionNode.from_json(data=data))
            i += 2
        else:
            merged_node_list.append(step)
            i += 1
    print(f"Double clicks merged: #{len(node_list)} -> #{len(merged_node_list)} steps.")
    return merged_node_list


# %% Time

def parse_time_from_path(path: str) -> float:
    """Parse the timestamp encoded at the beginning of the filename."""
    if path is None:
        raise ValueError("Cannot parse timestamp from None path.")
    filename = os.path.basename(str(path))
    match = re.search(r"(\d+\.\d+|\d+)", filename)
    if not match:
        raise ValueError(f"No numeric timestamp found in {path}")
    try:
        return float(match.group(1))
    except ValueError as exc:
        raise ValueError(f"Unable to parse timestamp from {path}") from exc


def safe_parse_timestamp(path: str | None) -> float | None:
    """Return the parsed timestamp or None if unavailable."""
    if not path:
        return None
    try:
        return parse_time_from_path(path)
    except (ValueError, TypeError):
        return None


def measure_time_from_states(states: list[dict], actions: list[str]) -> list[dict]:
    """Measure the time from the states."""
    time_list = []
    for i, state in enumerate(states):
        prev_after = time_list[-1]["after"] if time_list else None

        before_time = safe_parse_timestamp(state.get("before"))
        if before_time is None:
            before_time = safe_parse_timestamp(state.get("after"))
        if before_time is None:
            before_time = prev_after if prev_after is not None else 0.0

        after_time = safe_parse_timestamp(state.get("after"))
        if after_time is None and (i < len(states) - 1) and is_keyboard_action(actions[i]):
            after_time = safe_parse_timestamp(states[i+1].get("before"))
        if after_time is None:
            after_time = before_time

        time_range = max(0.0, after_time - before_time)

        if prev_after is None:
            time_diff = 0.0
        else:
            time_diff = before_time - prev_after
            if time_diff < 0:
                time_diff = 0.0

        time_list.append({
            "before": before_time, "after": after_time,
            "range": time_range, "diff": time_diff,
        })
    return time_list


def transfer_valid_states(node_list: list[ActionNode], src_suffix: str, dst_suffix: str):
    """Trasfer valid states to the new directory."""
    for i, node in enumerate(node_list):
        before_path, after_path = node.state.before, node.state.after
        if before_path is not None:
            dst_before_path = before_path.replace(src_suffix, dst_suffix)
            if os.path.exists(before_path):
                shutil.move(before_path, dst_before_path)
            node_list[i].state.before = dst_before_path
        
        if after_path is not None:
            dst_after_path = after_path.replace(src_suffix, dst_suffix)
            if os.path.exists(after_path):
                shutil.move(after_path, dst_after_path)
            node_list[i].state.after = dst_after_path

    return node_list


# %% Main
def main():
    # NEW APPROACH: Use screenshots as primary source of truth
    # 1. Load actions from screenshots (with accurate timestamps)
    screenshot_actions = load_actions_from_screenshots(args.screenshot_dir)
    
    # 2. Load DB actions to supplement with keypresses that don't have screenshots
    db_actions, db_timestamps_str, db_timestamps_sec = load_actions_from_db(args.data, args.db_path)
    print(f"Loaded {len(db_actions)} actions from the database.")
    
    # 3. Merge DB keypresses that don't have screenshots into the timeline
    all_actions = merge_db_keypresses_into_screenshot_actions(
        screenshot_actions,
        db_actions,
        db_timestamps_sec,
    )
    print(f"Combined action list: {len(all_actions)} actions.")
    
    # 3.5 Combine modifier key combos (optional)
    if True:
        all_actions = combine_modifier_key_combos(all_actions)
    
    # 4. Merge adjacent keyboard/scroll actions
    original_actions, merged_action_strings = merge_screenshot_actions(
        all_actions,
        enable_hotkey=True,
    )
    print(f"After merging: {len(original_actions)} actions.")
    
    # 5. Build states from the action dicts (paths already included)
    states = []
    for action_dict in original_actions:
        states.append({
            "before": action_dict["before_path"],
            "after": action_dict["after_path"],
        })
    
    # 6. Measure time from states
    time_list = measure_time_from_states(states, merged_action_strings)
    
    assert len(merged_action_strings) == len(states) == len(time_list)
    print(f"Original trajectory: #{len(merged_action_strings)} steps.")
    
    if args.adjust_states:
        states = adjust_states(merged_action_strings, states)
        # Recompute full time dicts after state adjustment
        time_list = measure_time_from_states(states, merged_action_strings)
    assert len(merged_action_strings) == len(states) == len(time_list)

    node_list = [ActionNode(action=a, state=s, time=t) for a, s, t in zip(merged_action_strings, states, time_list)]

    # organize trajectory
    if args.merge_double_clicks:
        node_list = merge_double_clicks(node_list)

    if args.transfer_valid_states:
        src_suffix = args.screenshot_dir.split('/')[-1]
        dst_suffix = args.state_dir.split('/')[-1]
        node_list = transfer_valid_states(node_list, src_suffix, dst_suffix)

    # Determine output path - organize into subdirectories
    if args.output_path:
        traj_path = os.path.abspath(os.path.expanduser(args.output_path))
        # If it's a directory, append the filename
        if os.path.isdir(traj_path) or (not os.path.exists(traj_path) and not traj_path.endswith('.json')):
            traj_path = os.path.join(traj_path, "processed_trajectory.json")
    else:
        # Default: save in 0_preprocessing subdirectory
        traj_dir = os.path.dirname(args.data)
        output_base = os.path.join(traj_dir, "0_preprocessing")
        os.makedirs(output_base, exist_ok=True)
        traj_path = os.path.join(output_base, "processed_trajectory.json")
    
    print(f"Saving trajectory of #{len(node_list)} steps to {traj_path}...")
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    root = SequenceNode(nodes=node_list)
    root.to_json(traj_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Directory containing screenshots and actions.db")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional path to save the processed trajectory JSON.")

    args = parser.parse_args()
    args.data = os.path.abspath(os.path.expanduser(args.data))
    args.db_path = os.path.join(args.data, "actions.db")
    args.screenshot_dir = os.path.join(args.data, "screenshots")
    if not os.path.isdir(args.screenshot_dir):
        raise FileNotFoundError(f"Screenshot directory not found: {args.screenshot_dir}")

    args.adjust_states = False
    args.merge_double_clicks = True
    args.transfer_valid_states = False
    args.state_dir = "states"
    args.enable_hotkey = True
    args.verbose = False

    main()
