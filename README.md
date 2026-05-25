# Offloading Score

Pipeline for inducing structured workflows from raw screen-and-keystroke recordings of a user working on a task, and calculating *offloading score*.

The pipeline takes a directory of raw recordings as input and produces:

- A segmented, labeled trajectory of what the user was doing at each moment
- A timeline with gaps closed, pauses detected, and remaining gaps classified
- Visualizations of the timeline and activity breakdown
- Per-step *output-use* labels and an aggregate offloading score for the session, by matching each step to a reference workflow

## Input: recording format

The pipeline expects a `records/` directory in the format produced by the **computer-recorder** in the [workflow-induction-toolkit](https://github.com/zorazrw/workflow-induction-toolkit/tree/main):

```
records/
├── actions.db
└── screenshots/
    ├── <timestamp>_<action>_first.jpg
    ├── <timestamp>_<action>_final.jpg
    ├── <timestamp>_<action>_before.jpg
    ├── <timestamp>_<action>_after.jpg
    └── ...
```

To generate recordings in this format, use the `computer-recorder` tool from the workflow-induction-toolkit repo linked above. It captures user activity on macOS / Windows into exactly this layout, so any recording produced there can be fed directly into this pipeline.

Optional inputs the pipeline will pick up automatically if present:

- A `.specstory/` directory adjacent to or inside `records/` — used to extract AI-assistant interaction logs.
- A `logs/` directory inside `records/` — fallback location for AI interaction logs.
- A parent directory containing multiple `issue_*` subfolders — used for data trimming across overlapping recordings and for fetching/summarizing GitHub issue context.

## Running the pipeline

```bash
python run_all.py --data /path/to/records
```

This runs all five stages in sequence. Outputs are written into sibling directories next to `records/` (i.e. into the parent of the `--data` directory):

```
<parent>/
├── records/                # input
├── 0_preprocessing/
│   └── processed_trajectory.json
├── 1_segment/
│   ├── segments.json
│   └── annotated.json
├── 2_induction/
│   ├── grouped.json
│   └── merged.json
├── 3_timeline/
│   ├── timeline.json
│   ├── 2_gaps/gaps_closed.json
│   ├── 3_pauses/with_pauses.json
│   └── 4_classified/classified.json
└── 5_visualizations/
```

### Resuming

If the pipeline fails partway through, re-run with `--resume` and it will detect existing intermediate outputs and skip completed stages:

```bash
python run_all.py --data /path/to/records --resume
```

### Skipping stages individually

```bash
python run_all.py --data /path/to/records \
  --skip-preprocessing \
  --skip-segmentation \
  --skip-induction \
  --skip-timeline \
  --skip-visualization
```

Other useful flags:

- `--threshold <float>` — fixed MSE threshold for segmentation (default: derived from `--auto-percentile`, default 75).
- `--no-remerge` — disable LLM-based re-merging of adjacent segments.
- `--min-gap <seconds>` — minimum gap length to keep when closing gaps in the timeline (default 5.0).
- `--issues-root <path>` — directory of `issue_*` folders for data trimming and issue summarization.
- `--verbose` — print detailed progress for each stage.

## Pipeline stages

| Stage | Directory | What it does |
|-------|-----------|--------------|
| 0. Preprocessing | `0_preprocessing/` | Optionally trim overlapping recordings; extract a processed trajectory from `actions.db` + screenshots; extract AI interactions from logs; optionally summarize a linked GitHub issue. |
| 1. Segmentation | `1_segment/` | Split the trajectory into segments by visual change, then annotate each segment with an activity description. |
| 2. Induction | `2_induction/` | Induce high-level activity labels (`labels.txt`) over the annotated segments and merge adjacent segments that share a label. |
| 3. Timeline | `3_timeline/` | Map merged segments onto a wall-clock timeline, close small gaps, insert detected pauses, and classify remaining gaps using `gap_labels.txt`. |
| 4. Visualization | `4_visualize/` | Render timeline and breakdown charts into `5_visualizations/`. |
| 5. Scoring | `5_scoring/` | Match each non-human step to a reference workflow and apply per-step *output-use* labels used to compute the offloading score. |

## Computing the offloading score

Once `run_all.py` has produced the per-step annotations for a session, the offloading score is computed by matching each non-human step in the recording to a reference workflow and applying *output-use* labels.

The reference workflow is generated separately, from the same recording, by the [workflow-induction-toolkit](https://github.com/zorazrw/workflow-induction-toolkit/tree/main). That toolkit's `induce_workflow.sh` produces two artifacts you need here:

- `workflow.json` — structured workflow representation
- `workflow.txt` — flat, line-per-step natural-language version of the workflow

### Step 1: Convert the workflow JSON into block-level actor turns

```bash
python 5_scoring/parse_json_to_turns.py --input-json /path/to/workflow.json
```

Writes `/path/to/workflow_annotated.json` by default. Optional flags:

```bash
python 5_scoring/parse_json_to_turns.py \
  --input-json /path/to/workflow.json \
  --output-json /path/to/custom_annotated.json \
  --print-tree
```

### Step 2: Label turns against the workflow and compute offloading

```bash
python 5_scoring/analyze_jsons_with_workflow_context.py \
  --annotated-json /path/to/workflow_annotated.json \
  --workflow-txt   /path/to/workflow.txt \
  --time-split     long \
  --task-description "Build a local web app that ..." \
  --task-id        timer_app
```

Or pass the task description as a file:

```bash
python 5_scoring/analyze_jsons_with_workflow_context.py \
  --annotated-json /path/to/workflow_annotated.json \
  --workflow-txt   /path/to/workflow.txt \
  --time-split     long \
  --task-description-file /path/to/task.txt \
  --task-id        timer_app
```

Required flags:

- `--annotated-json` — the `_annotated.json` from `parse_json_to_turns.py`.
- `--workflow-txt` — flat workflow text from the workflow-induction-toolkit.
- `--time-split {long,short}` — label indicating whether this run is from the long or short time split.
- Exactly one of `--task-description TEXT` or `--task-description-file PATH` — free-text description of what the user was trying to do.

Optional flags:

- `--task-id ID` — free-form identifier recorded in the output JSON for bookkeeping (default: `task`).
- `--input-json` — raw merged JSON, used only to compute `time_taken` from top-level `time.start` / `time.end`.
- `--output-json` — explicit output path. Defaults to replacing `_annotated.json` with `_annotated_labeled_workflow_context.json`.
- `--output-use-only` — compute only the workflow matching + output-use labels; skip rewrite/counterfactual and cognitive annotations.

The output JSON contains per-step output-use labels and the matched workflow indices, from which the aggregate offloading score for the session is computed.

## Requirements

- Python 3.10+
- An OpenAI API key in the environment (`OPENAI_API_KEY`) — used by the segmentation, annotation, induction, and gap-classification stages.

## Related

- [workflow-induction-toolkit](https://github.com/zorazrw/workflow-induction-toolkit/tree/main) — the source of the recording format and the `computer-recorder` tool used to produce inputs for this pipeline.
