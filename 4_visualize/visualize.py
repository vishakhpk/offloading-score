"""Visualize developer activity timeline from timeline JSON files."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure matplotlib uses non-interactive backend
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Ensure project root is importable for utils
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    pstr = str(path)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)

from utils import load_json

# Activity color mapping
# - Reading family: cool blues/teals (grouped similar)
# - Writing family: warm reds/pinks (grouped similar, distinct from reading)
# - Testing family: yellows/oranges (grouped similar, distinct from others)
ACTIVITY_COLORS: Dict[str, str] = {
    # Reading
    "reading code": "#2F80ED",
    "reading docs": "#2D9CDB",
    "reading issue": "#56CCF2",
    "reading generation": "#6FCF97",

    # Writing
    "writing code": "#EB5757",
    "writing docs": "#8E44AD",
    "writing prompt": "#F2C94C",
    "ai code cleaning": "#E57373",
    "generation taken": "#FF8A80",
    "generation rejected": "#D81B60",

    # Testing
    "writing tests": "#F2C94C",
    "test running tests": "#F2994A",
    "test manually checking": "#F9A825",

    # Other activities
    "compiling": "#20B2AA",
    "running debugger": "#FF1493",
    "debugging": "#DDA15E",
    "troubleshooting": "#BC6C25",
    "git": "#417505",
    "branching": "#4A7023",
    "committing": "#5C8B3E",
    "pr": "#6FA65A",
    "setup": "#95A5A6",
    "communicating with teammates": "#E74C3C",
    "planning": "#FFFFBA",
    "browsing": "#90E0EF",
    "searching": "#FFB3BA",
    "reviewing": "#BAFFC9",
    "waiting on generation": "#9B59B6",
    "thinking": "#CFBAF0",
    "unrelated": "#BDC3C7",
    "unknown": "#FFFFFF",
    "pause": "#FFFFFF",
    "broken": "#34495E",
    "replicating bug": "#C0392B",
    "screenshot": "#D3D3D3",
}

# ACTIVITY_COLORS = {
#     "reading code": "#4A90E2",
#     "reading docs": "#7ED321",
#     "reading issue": "#50E3C2",
#     "writing prompt": "#F5A623",
#     "reading generation": "#BD10E0",
#     "generation taken": "#9013FE",
#     "generation rejected": "#8B572A",
#     "ai code cleaning": "#B8E986",
#     "writing code": "#D0021B",
#     "writing tests": "#F8E71C",
#     "test running tests": "#FF6B6B",
#     "test manually checking": "#FFA07A",
#     "compiling": "#20B2AA",
#     "running debugger": "#FF1493",
#     "git": "#417505",
#     "branching": "#4A7023",
#     "committing": "#5C8B3E",
#     "pr": "#6FA65A",
#     "setup": "#95A5A6",
#     "communicating with teammates": "#E74C3C",
#     "unrelated": "#BDC3C7",
#     "unknown": "#FFFFFF",
#     "pause": "#FFFFFF",
#     "broken": "#34495E",
#     "waiting on generation": "#9B59B6",
#     "writing docs": "#16A085",
#     "replicating bug": "#C0392B"
# }



def resolve_activity_color(activity: str, fallback_colors: Dict[str, str], palette: List[str]) -> str:
    """Resolve color for an activity, using predefined colors or generating from palette."""
    # Paused activities are always white
    if activity.lower() == "pause":
        return "#FFFFFF"
    if activity in ACTIVITY_COLORS:
        return ACTIVITY_COLORS[activity]
    if activity not in fallback_colors:
        color_index = len(fallback_colors) % len(palette)
        fallback_colors[activity] = palette[color_index]
    return fallback_colors[activity]


def plot_horizontal_timeline(
    timeline: List[Dict],
    output_path: Path,
    palette: Optional[List[str]] = None,
    color_lookup: Optional[Dict[str, str]] = None,
) -> None:
    """Plot a horizontal timeline from timeline entries.
    
    Args:
        timeline: List of timeline entries with start_time, end_time, duration, label
        output_path: Path to save the visualization
        palette: Optional color palette for activities not in ACTIVITY_COLORS
        color_lookup: Optional pre-computed color lookup dictionary
    """
    if not timeline:
        print("No timeline entries to plot.")
        return
    
    # Sort timeline by start_time
    timeline = sorted(timeline, key=lambda x: x["start_time"])
    start_time = timeline[0]["start_time"]
    
    # Calculate offsets and durations in minutes
    offsets: List[float] = []
    durations: List[float] = []
    
    for entry in timeline:
        offsets.append((entry["start_time"] - start_time) / 60.0)
        durations.append(entry["duration"] / 60.0)
    
    bar_height = 0.8
    fig, ax = plt.subplots(figsize=(18, 3.75))
    palette = palette or [plt.get_cmap("tab20")(idx) for idx in range(20)]
    fallback_colors = color_lookup if color_lookup is not None else {}
    total_width = max(offset + duration for offset, duration in zip(offsets, durations))
    
    # Plot bars for each timeline entry
    for offset, duration, entry in zip(offsets, durations, timeline):
        activity = entry.get("label", "unknown")
        color = resolve_activity_color(activity, fallback_colors, palette)
        # Use edgecolor for white/paused activities to make them visible
        # edgecolor = "gray" if activity.lower() == "pause" else "none"
        edgecolor = "none"
        ax.barh(
            0,
            duration,
            left=offset,
            height=bar_height,
            color=color,
            edgecolor=edgecolor,
            linewidth=0.5 if activity.lower() == "pause" else 0.0,
            alpha=0.88,
        )
    
    ax.set_xlim(0, total_width * 1.01)
    ax.set_xlabel("Minutes from first activity")
    ax.set_title("Developer Activity Timeline", fontsize=14, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-bar_height, bar_height)
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    
    # Create legend
    legend_handles = []
    for activity in sorted({entry.get("label", "unknown") for entry in timeline}):
        legend_handles.append(
            mpatches.Patch(
                color=resolve_activity_color(activity, fallback_colors, palette),
                label=activity
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9,
            framealpha=0.9
        )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved horizontal timeline to {output_path}")


def plot_activity_breakdown(
    timeline: List[Dict],
    output_path: Path,
    *,
    included_activities: Optional[set] = None,
    palette: Optional[List[str]] = None,
    color_lookup: Optional[Dict[str, str]] = None,
) -> None:
    """Plot total duration by activity as a horizontal bar chart.
    
    Args:
        timeline: List of timeline entries
        output_path: Path to save the visualization
        included_activities: Optional set of activities to include
        palette: Optional color palette
        color_lookup: Optional pre-computed color lookup dictionary
    """
    if not timeline:
        print("No timeline entries to plot.")
        return
    
    # Calculate totals by activity (excluding pause)
    totals: Dict[str, float] = defaultdict(float)
    for entry in timeline:
        activity = entry.get("label", "unknown")
        # Skip pause activities in breakdown
        if activity.lower() == "pause":
            continue
        if included_activities is not None and activity not in included_activities:
            continue
        totals[activity] += entry.get("duration", 0.0)
    
    if not totals:
        print("No activities remain after filtering; skipping breakdown plot.")
        return
    
    # Sort by duration (descending)
    ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in ordered]
    durations = [seconds / 60.0 for _, seconds in ordered]
    
    palette = palette or [plt.get_cmap("tab20")(idx) for idx in range(20)]
    fallback_colors = color_lookup if color_lookup is not None else {}
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
    colors = [resolve_activity_color(activity, fallback_colors, palette) for activity in labels]
    ax.barh(labels, durations, color=colors, alpha=0.9)
    ax.set_xlabel("Total duration (minutes)")
    ax.set_title("Total Duration by Activity", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    
    # Add duration labels on bars
    for y, duration in enumerate(durations):
        ax.text(duration + 0.5, y, f"{duration:.1f} min", va="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved activity breakdown to {output_path}")


def calculate_statistics(timeline: List[Dict]) -> None:
    """Calculate and print statistics about the timeline.
    
    Args:
        timeline: List of timeline entries
    """
    if not timeline:
        print("No timeline entries to analyze.")
        return
    
    total_duration = sum(entry.get("duration", 0.0) for entry in timeline)
    activity_counts = Counter(entry.get("label", "unknown") for entry in timeline)
    activity_durations: Dict[str, float] = defaultdict(float)
    
    for entry in timeline:
        activity = entry.get("label", "unknown")
        activity_durations[activity] += entry.get("duration", 0.0)
    
    print(f"\n--- Timeline Statistics ---")
    print(f"Total entries: {len(timeline)}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min, {total_duration/3600:.2f} hours)")
    print(f"\nActivity counts:")
    for activity, count in activity_counts.most_common():
        duration = activity_durations[activity]
        percentage = (duration / total_duration * 100) if total_duration > 0 else 0.0
        print(f"  {activity:30s}: {count:4d} entries, {duration/60:7.2f} min ({percentage:5.1f}%)")


def compress_timeline(
    timeline: List[Dict],
    *,
    compress_pause: bool = False,
    pause_compress_seconds: float = 0.0,
) -> List[Dict]:
    """Reflow timeline so large pauses don't dominate the x-axis.

    When enabled, rebuilds start/end/duration using the original order and segment
    durations, optionally shrinking pauses to a small (or zero) duration.
    """
    if not timeline:
        return timeline

    # Ensure chronological order
    sorted_tl = sorted(timeline, key=lambda x: x["start_time"])
    current_time = 0.0
    compressed: List[Dict] = []

    for entry in sorted_tl:
        label = str(entry.get("label", "")).lower()
        original_duration = entry.get("duration")
        if original_duration is None:
            original_duration = entry.get("end_time", 0) - entry.get("start_time", 0)
        duration = float(original_duration)

        if compress_pause and label == "pause":
            duration = min(max(pause_compress_seconds, 0.0), duration)

        new_entry = dict(entry)
        new_entry["start_time"] = current_time
        new_entry["end_time"] = current_time + duration
        new_entry["duration"] = duration

        compressed.append(new_entry)
        current_time += duration

    return compressed


def main():
    parser = argparse.ArgumentParser(
        description="Visualize developer activity timeline from timeline JSON file."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to timeline JSON file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store generated visualizations. Defaults to same directory as input file."
    )
    parser.add_argument(
        "--skip-breakdown",
        action="store_true",
        help="Skip generating the total-duration-by-activity bar chart."
    )
    parser.add_argument(
        "--activities",
        nargs="*",
        default=None,
        help="List of activities to include. If not specified, includes all activities."
    )
    parser.add_argument(
        "--compress-pauses",
        action="store_true",
        help="Reflow timeline so pauses are compressed and don't stretch the x-axis.",
    )
    parser.add_argument(
        "--pause-compress-seconds",
        type=float,
        default=0.0,
        help="Duration to use for compressed pauses when --compress-pauses is set (default: 0 = remove pause duration).",
    )
    args = parser.parse_args()
    
    # Load timeline
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find timeline file at {data_path}")
    
    timeline = load_json(data_path)
    
    if not timeline:
        print("No timeline entries found in file.")
        return
    
    print(f"Loaded {len(timeline)} timeline entries from {data_path}")
    
    # Filter activities if specified
    included_activities = None
    if args.activities:
        included_activities = set(args.activities)
        print(f"Including activities: {', '.join(sorted(included_activities))}")
        timeline = [entry for entry in timeline if entry.get("label") in included_activities]
        print(f"Filtered to {len(timeline)} entries")
    
    # Calculate statistics
    # Optionally compress pauses to avoid dwarfing other segments
    if args.compress_pauses:
        timeline = compress_timeline(
            timeline,
            compress_pause=True,
            pause_compress_seconds=args.pause_compress_seconds,
        )
        print(
            f"Compressed timeline with pauses set to {args.pause_compress_seconds:.2f}s "
            f"(pauses remain as entries but with reduced duration)."
        )

    calculate_statistics(timeline)
    
    # Determine output directory - organize into subdirectories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Check if we're in a numbered subdirectory (1_timeline, 2_gaps, etc.)
        parent_name = data_path.parent.name
        if parent_name.startswith(("1_", "2_", "3_", "4_", "5_")):
            # Find the issue directory (walk up past processing step directories)
            current = data_path.parent
            while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified", "5_visualizations"]:
                current = current.parent
            output_dir = current / "4_visualizations"
        else:
            # Find the issue directory (walk up past processing step directories)
            current = data_path.parent
            while current.name in ["2_induction", "1_segment", "0_preprocessing", "3_timeline", "2_gaps", "3_pauses", "4_classified", "5_visualizations"]:
                current = current.parent
            output_dir = current / "5_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare color palette and lookup
    palette = [plt.get_cmap("tab20")(idx) for idx in range(20)]
    shared_colors: Dict[str, str] = {}
    for entry in timeline:
        activity = entry.get("label", "unknown")
        if included_activities is None or activity in included_activities:
            resolve_activity_color(activity, shared_colors, palette)
    
    # Generate visualizations with shorter names
    suffix = "_".join(sorted(included_activities)) if included_activities else "all"
    timeline_path = output_dir / f"timeline_{suffix}.png"
    breakdown_path = output_dir / f"breakdown_{suffix}.png"
    
    plot_horizontal_timeline(
        timeline,
        timeline_path,
        palette=palette,
        color_lookup=shared_colors,
    )
    
    if not args.skip_breakdown:
        plot_activity_breakdown(
            timeline,
            breakdown_path,
            included_activities=included_activities,
            palette=palette,
            color_lookup=shared_colors,
        )
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

