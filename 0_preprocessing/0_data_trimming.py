"""Trim overlapping issue folders so each only keeps data after the previous issue's timeline."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# Initialize package (sets up sys.path)
_init_path = Path(__file__).parent / "__init__.py"
exec(_init_path.read_text())

from dotenv import load_dotenv

# Supabase is an optional dependency used only for the (private) data-trimming
# code path that fetches issue metadata. If it's not installed, the trimming
# step is silently skipped (the guard at the call site checks `create_client`).
try:
    from supabase import create_client
except ImportError:
    create_client = None

from utils import load_json, save_json

load_dotenv()

def _find_paths(issue_dir: Path, *rel_paths: str) -> list[Path]:
    """Find existing paths from a list of relative paths."""
    return [p for p in [issue_dir / path for path in rel_paths] if p.exists()]


def _timestamp_from_filename(path: Path) -> float | None:
    """Extract timestamp from filename (assumes format: timestamp_rest)."""
    parts = path.name.split("_", 1)
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _latest_actions_db_time(issue_dir: Path) -> float | None:
    """Get the latest timestamp from actions.db files."""
    latest: float | None = None
    for db_path in _find_paths(issue_dir, "data/actions.db", "actions.db"):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(strftime('%s', created_at)) FROM observations")
            row = cur.fetchone()
            if row and row[0] is not None:
                val = float(row[0])
                latest = max(latest, val) if latest is not None else val
    return latest


def _latest_screenshot_time(issue_dir: Path) -> float | None:
    """Get the latest timestamp from screenshot files."""
    latest: float | None = None
    for directory in _find_paths(issue_dir, "data/screenshots", "screenshots"):
        if not directory.is_dir():
            continue
        for file in directory.iterdir():
            if file.is_file() and (ts := _timestamp_from_filename(file)) is not None:
                latest = max(latest, ts) if latest is not None else ts
    return latest


def trim_screenshots(issue_dir: Path, min_time: float | None, max_time: float | None = None) -> None:
    """Remove screenshots outside the time window."""
    for directory in _find_paths(issue_dir, "data/screenshots", "screenshots"):
        if not directory.is_dir():
            continue
        removed = kept = 0
        for file in directory.iterdir():
            if not file.is_file():
                continue
            ts = _timestamp_from_filename(file)
            if ts is None:
                kept += 1
            elif (min_time is not None and ts <= min_time) or (max_time is not None and ts >= max_time):
                file.unlink()
                removed += 1
            else:
                kept += 1
        print(f"Screenshots in {directory}: kept {kept}, removed {removed} outside [{min_time}, {max_time}]")


def trim_actions_db(issue_dir: Path, min_time: float | None, max_time: float | None = None) -> None:
    """Remove database rows outside the time window."""
    for db_path in _find_paths(issue_dir, "data/actions.db", "actions.db"):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            clauses, params = [], []
            if min_time is not None:
                clauses.append("strftime('%s', created_at) <= ?")
                params.append(min_time)
            if max_time is not None:
                clauses.append("strftime('%s', created_at) >= ?")
                params.append(max_time)
            if not clauses:
                continue
            where = " OR ".join(clauses)
            cur.execute(f"SELECT COUNT(*) FROM observations WHERE {where}", params)
            to_delete = cur.fetchone()[0]
            cur.execute(f"DELETE FROM observations WHERE {where}", params)
            conn.commit()
            print(f"Trimmed {to_delete} rows from {db_path} outside window [{min_time}, {max_time}]")


def trim_issue(issue_dir: Path, min_time: float | None, max_time: float | None = None) -> None:
    """Trim screenshots and actions.db for a single issue directory."""
    trim_screenshots(issue_dir, min_time, max_time)
    trim_actions_db(issue_dir, min_time, max_time)


def _get_issue_end_time(issue_dir: Path) -> float | None:
    """Get the latest timestamp from screenshots or actions.db."""
    times = [t for t in [_latest_screenshot_time(issue_dir), _latest_actions_db_time(issue_dir)] if t is not None]
    return max(times) if times else None


def _coerce_epoch(value) -> float | None:
    """Convert ISO datetime string to epoch timestamp.
    """

    # Strip whitespace
    normalized = value.strip()

    # Replace Z with +00:00 for Python's fromisoformat
    if normalized.endswith('Z'):
        normalized = normalized[:-1] + '+00:00'

    # Extract timezone suffix if present
    tz_match = re.search(r'([+-]\d{2}:\d{2})$', normalized)
    tz_suffix = tz_match.group(1) if tz_match else ''
    main_part = normalized[:-len(tz_suffix)] if tz_suffix else normalized

    # Normalize fractional seconds to exactly 6 digits
    if '.' in main_part:
        date_part, fractional = main_part.rsplit('.', 1)
        # Keep only the digits from the fractional part
        fractional_digits = ''.join(c for c in fractional if c.isdigit())
        # Pad or truncate to 6 digits
        if len(fractional_digits) > 6:
            fractional_digits = fractional_digits[:6]
        else:
            fractional_digits = fractional_digits.ljust(6, '0')
        normalized = f"{date_part}.{fractional_digits}{tz_suffix}"

    # Parse and convert to UTC timestamp
    dt = _dt.datetime.fromisoformat(normalized)
    return dt.astimezone(_dt.timezone.utc).timestamp()


def _extract_issue_id(name: str, prefix: str = "issue") -> str:
    """Extract issue ID from folder name."""
    if name.lower().startswith(prefix.lower()):
        return name[len(prefix):].lstrip(" _-") or name
    return name


def _find_issue_dir(issue_id: str, issues_root: Path, prefix: str = "issue") -> Path | None:
    """Find issue directory by ID."""
    issues_root = issues_root.expanduser().resolve()
    for child in issues_root.iterdir():
        if child.is_dir() and _extract_issue_id(child.name, prefix) == issue_id:
            return child
    return None


def _fetch_issue_metadata(client, table: str, issue_id: str) -> dict | None:
    """Fetch all issue metadata from Supabase in one query."""
    result = client.table(table).select("*").eq("issue_id", issue_id).limit(1).execute()
    data = getattr(result, "data", None) or []
    return data[0] if data else None


def _write_issue_metadata_file(issue_dir: Path, entry: dict) -> None:
    """Write issue metadata to issue.json file in the issue directory."""
    metadata_path = issue_dir / "issue.json"
    # Merge with existing metadata if present
    existing = entry.copy()
    if metadata_path.exists():
        try:
            existing_data = load_json(metadata_path)
            # Preserve existing fields not in entry
            for key, value in existing_data.items():
                if key not in existing:
                    existing[key] = value
        except (OSError, json.JSONDecodeError):
            pass
    
    save_json(existing, metadata_path)
    print(f"Wrote metadata to {metadata_path}")


def _discover_issue_dirs(root: Path, require_timing: bool = True) -> list[Path]:
    """Discover all valid issue directories."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Issues directory not found: {root}")
    issue_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.lower().startswith("issue"):
            continue
        if require_timing:
            end_time = _get_issue_end_time(child)
            if end_time is None:
                print(f"Skipping {child.name}: no timing data (screenshots or actions.db) found.")
                continue
        issue_dirs.append(child)
    return issue_dirs


def _write_issue_metadata_from_supabase(issues_root: Path, issue_dirs: list[Path]) -> None:
    """Fetch issue metadata from Supabase and write issue.json files."""
    supabase_url = os.getenv("SUPABASE_DEV_URL")
    supabase_key = os.getenv("SUPABASE_DEV_KEY")
    client = create_client(supabase_url, supabase_key)
    
    table, prefix = "repo-issues", "issue"
    issue_ids = [_extract_issue_id(p.name, prefix) for p in issue_dirs]
    
    for issue_id in issue_ids:
        issue_dir = _find_issue_dir(issue_id, issues_root, prefix)
        if not issue_dir:
            print(f"{issue_id}: no matching folder, skipping.")
            continue
        
        record = _fetch_issue_metadata(client, table, issue_id)
        if not record:
            print(f"{issue_id}: no Supabase rows found, skipping.")
            continue
        
        # Write metadata to individual issue folder
        entry = {"issue_id": issue_id}
        entry["issue_url"] = record.get("issue_url")
        entry["using_ai"] = record.get("using_ai")
        _write_issue_metadata_file(issue_dir, entry)


def _trim_with_supabase(issues_root: Path, issue_dirs: list[Path]) -> None:
    """Trim issues using Supabase timestamps."""
    client = create_client(os.getenv("SUPABASE_DEV_URL"), os.getenv("SUPABASE_DEV_KEY"))
    table, prefix = "repo-issues", "issue"
    issue_ids = [_extract_issue_id(p.name, prefix) for p in issue_dirs]
    
    for issue_id in issue_ids:
        issue_dir = _find_issue_dir(issue_id, issues_root, prefix)
        if not issue_dir:
            continue
        
        record = _fetch_issue_metadata(client, table, issue_id)
        if not record:
            continue
        
        start_epoch = _coerce_epoch(record.get("accepted_on"))
        end_epoch = _coerce_epoch(record.get("completed_on"))
        if start_epoch or end_epoch:
            print(f"{issue_id}: trimming to window [{start_epoch}, {end_epoch}] in {issue_dir}")
            trim_issue(issue_dir, start_epoch, end_epoch)


def _trim_with_overlap(issue_dirs: list[Path]) -> None:
    """Trim issues using overlap detection from local timestamps."""
    issue_times = [(end_time, issue) for issue in issue_dirs if (end_time := _get_issue_end_time(issue)) is not None]
    if len(issue_times) < 2:
        raise SystemExit("Not enough issues with timing data to trim.")
    
    issue_times.sort(key=lambda x: x[0])
    print("Issues ordered by end time (oldest -> newest):")
    for max_time, issue in issue_times:
        print(f"- {issue} ends at {max_time}")
    
    cutoff = issue_times[0][0]
    for max_time, issue in issue_times[1:]:
        print(f"Trimming {issue.name} so it only keeps events after {cutoff} (previous issue end).")
        trim_issue(issue, cutoff, None)
        cutoff = max(cutoff, max_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim issue folders using Supabase timestamps when available; otherwise fall back to overlap trimming."
    )
    parser.add_argument("--data", type=Path, required=True, help="Parent directory containing issue_* folders.")
    args = parser.parse_args()
    
    issues_root = args.data
    
    # Always try to write issue.json files from Supabase, even if trimming fails
    all_issue_dirs = list(dict.fromkeys(_discover_issue_dirs(issues_root, require_timing=False)))
    print(f"Discovered {len(all_issue_dirs)} issue folder(s): {[d.name for d in all_issue_dirs]}")
    _write_issue_metadata_from_supabase(issues_root, all_issue_dirs)
    
    # Discover issue dirs with timing data for trimming
    issue_dirs = list(dict.fromkeys(_discover_issue_dirs(issues_root, require_timing=True)))
    print(f"Found {len(issue_dirs)} issue folder(s) with timing data: {[d.name for d in issue_dirs]}")
    
    # Try Supabase-based trimming first (works with any number of folders)
    if create_client and os.getenv("SUPABASE_DEV_URL") and os.getenv("SUPABASE_DEV_KEY"):
        if issue_dirs:
            _trim_with_supabase(issues_root, issue_dirs)
        else:
            print("No issue folders with timing data found; skipping trimming.")
    else:
        print("Skipping Supabase-based trim: missing credentials or supabase client import.")
    
    # Fallback to overlap-based trimming (requires at least 2 folders)
    if len(issue_dirs) >= 2:
        _trim_with_overlap(issue_dirs)
    elif len(issue_dirs) == 1:
        print(f"Overlap trimming requires at least 2 folders with timing data; skipping.")


if __name__ == "__main__":
    main()
