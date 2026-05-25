from __future__ import annotations

import argparse
import json
import os
import sys
from textwrap import dedent
from typing import Any, Dict, List, Optional


# The task description is supplied at runtime via --task-description (a string)
# or --task-description-file (a path to a text file). See the CLI in main().


def _get_openai_client(client: Optional[object] = None):
    """Create an OpenAI client lazily so argparse --help works without openai installed."""
    if client is not None:
        return client
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required to run analysis. Install it in your environment."
        ) from exc
    return OpenAI()


def llm_match_step_to_workflow(
    current_ai_assisted_step: str,
    workflow: List[str],
    *,
    previous_human_step: Optional[str] = None,
    previous_step: Optional[Dict[str, Any]] = None,
    next_step: Optional[Dict[str, Any]] = None,
    previous_matched_index: Optional[int] = None,
    merged_hints: Optional[List[str]] = None,
    model: str = "gpt-4.1-mini",
    client: Optional[object] = None,
) -> int:
    """Match a target AI-assisted step to exactly one workflow item."""
    if not current_ai_assisted_step.strip():
        raise ValueError("`current_ai_assisted_step` must be a non-empty string.")
    if not workflow or not all(isinstance(x, str) for x in workflow):
        raise ValueError("`workflow` must be a non-empty list of strings.")

    client = _get_openai_client(client)

    def step_to_text(step: Optional[Dict[str, Any]]) -> str:
        if not step:
            return "N/A"
        actor = step.get("actor", "")
        prefix = "Human" if actor == "Human" else (actor or "Unknown")
        return f"{prefix}: {step.get('text', '')}".strip()

    workflow_windows = []
    for i, workflow_step in enumerate(workflow):
        prev_workflow = workflow[i - 1] if i > 0 else "N/A"
        next_workflow = workflow[i + 1] if i + 1 < len(workflow) else "N/A"
        workflow_windows.append(
            "\n".join(
                [
                    f"[{i}]",
                    f"Previous Workflow Step: {prev_workflow}",
                    f"Candidate Workflow Step: {workflow_step}",
                    f"Next Workflow Step: {next_workflow}",
                ]
            )
        )
    workflow_lines = "\n\n".join(workflow_windows)

    instructions = (
        "You are matching a current AI-assisted step to one workflow item.\n"
        "Choose EXACTLY ONE workflow item.\n\n"
        "Rules:\n"
        "- Return ONLY a JSON object\n"
        "- The JSON must have only one key \"index\" which is an integer\n"
        "- index must be between 0 and {max_idx}\n"
        "- Do not return text, explanations, or multiple indices\n"
        "- Use the local step context and the workflow windows to find the closest semantic match\n"
        "- Favor the candidate whose surrounding workflow context best matches the local action context\n"
        "- Treat the previous matched workflow index as a very weak tie-breaker only\n"
        "- If the semantic match points elsewhere, ignore the previous matched workflow index hint\n"
    ).format(max_idx=len(workflow) - 1)

    previous_match_hint = (
        str(previous_matched_index) if previous_matched_index is not None else "N/A"
    )
    user_input = "\n".join(
        [
            "LOCAL ACTION CONTEXT:",
            f"Previous Human Step: {previous_human_step or 'N/A'}",
            f"Previous Raw Step: {step_to_text(previous_step)}",
            f"Current AI-Assisted Step: {current_ai_assisted_step}",
            f"Next Raw Step: {step_to_text(next_step)}",
            f"Previous Matched Workflow Index Hint (weak tie-breaker only): {previous_match_hint}",
            "Nearby merged.json hints: "
            + (" || ".join(merged_hints) if merged_hints else "N/A"),
            "",
            "WORKFLOW CANDIDATES:",
            workflow_lines,
        ]
    )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )

    text = (resp.output_text or "").strip()
    if not text:
        raise RuntimeError("Empty model output.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON.\nRaw output:\n{text}") from exc

    idx = parsed.get("index")
    if not isinstance(idx, int) or not (0 <= idx < len(workflow)):
        raise RuntimeError(f"Invalid index returned: {idx}")
    return idx


def get_idx_from_filepath(filepath: str) -> int:
    basename = os.path.basename(filepath)
    basename = os.path.splitext(basename)[0]
    parts = basename.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def extract_time_taken(input_json_path: Optional[str]) -> Optional[float]:
    if not input_json_path:
        return None
    with open(input_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        time_info = raw_data.get("time", {})
        start = time_info.get("start")
        end = time_info.get("end")
        if start is None or end is None:
            return None
        return end - start
    if isinstance(raw_data, list):
        total = 0.0
        found_any = False
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            time_info = item.get("time", {})
            start = time_info.get("start")
            end = time_info.get("end")
            if start is None or end is None:
                continue
            total += end - start
            found_any = True
        return total if found_any else None
    return None


def should_use_merged_hints(current_ai_assisted_step: str) -> bool:
    text = current_ai_assisted_step.lower()
    generic_markers = [
        "displayed",
        "provided/generated",
        "guidance",
        "instructions",
        "side panel",
        "panel",
        "feedback",
        "suggestions",
        "code/instruction",
    ]
    return any(marker in text for marker in generic_markers)


def merged_hint_text(merged_item: Dict[str, Any]) -> str:
    parts = []
    label = merged_item.get("label")
    annotation = merged_item.get("annotation")
    source_indices = merged_item.get("source_indices")
    if label:
        parts.append(f"label={label}")
    if annotation:
        parts.append(f"annotation={annotation}")
    if source_indices:
        parts.append(f"source_indices={source_indices}")
    return " | ".join(parts)


def top_merged_hints(
    merged_data: Optional[Any],
    *,
    current_ai_assisted_step: str,
    previous_human_step: Optional[str],
    previous_step: Optional[Dict[str, Any]],
    next_step: Optional[Dict[str, Any]],
    limit: int = 3,
) -> List[str]:
    if not isinstance(merged_data, list):
        return []

    def tokenize(text: str) -> set[str]:
        tokens = []
        current = []
        for ch in text.lower():
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return {token for token in tokens if len(token) >= 4}

    query_parts = [current_ai_assisted_step, previous_human_step or ""]
    if previous_step:
        query_parts.append(previous_step.get("text", ""))
    if next_step:
        query_parts.append(next_step.get("text", ""))
    query_tokens = tokenize(" ".join(query_parts))
    if not query_tokens:
        return []

    scored: List[tuple[int, int, str]] = []
    for idx, item in enumerate(merged_data):
        if not isinstance(item, dict):
            continue
        hint = merged_hint_text(item)
        if not hint:
            continue
        overlap = query_tokens & tokenize(hint)
        if not overlap:
            continue
        scored.append((len(overlap), idx, hint))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [f"merged[{idx}] overlap={score}: {hint}" for score, idx, hint in scored[:limit]]


def build_rewrite_prompt(previous_turns, tool_turn, next_turn):
    context_block = json.dumps(previous_turns, indent=2)
    tool_block = json.dumps(tool_turn, indent=2)
    next_block = json.dumps(next_turn, indent=2) if next_turn else "null"

    return dedent(
        f"""
    You are rewriting a single step in a human-AI collaborative coding action with a
    realistic user-only steps. Essentially, you are estimating how a human would have
    accomplished the same step without any AI assistance, while matching the granularity of
    surrounding user actions. You can introduce any additional tools the user might have used
    to accomplish the task in the absence of AI. Your goal is to replace the single human-AI step
    with an appropriate number of user-only steps that accomplish the same code changes or actions.

    GOAL:
    Replace all AI actions that occur in the current TOOL turn with an appropriate number
    of USER ONLY turns that:
    - Accomplish the same code changes or actions performed that the human-AI step did
    - Match the granularity of surrounding workflow steps
    - Are realistic human workflow actions
    - Do NOT include any AI actions or references to AI assistance
    - Introduce any additional tools they might have used to accomplish the task in the absence of AI
    - Are NOT too small (no single keystrokes)
    - Are NOT too large (no "wrote entire file")

    CONTEXT — PREVIOUS TURNS:
    {context_block}

    TOOL TURN TO REWRITE:
    {tool_block}

    NEXT TURN (for alignment):
    {next_block}

    OUTPUT FORMAT:
    - Return ONLY the reconstructed steps, a list of strings which are the steps that would replace
      the single human-AI step if the user did not have access to any AI tools.
    - Do NOT include explanations.
    - Do NOT include markdown.
    """
    )


def rewrite_tool_turn(previous_turns, tool_turn, next_turn, client=None, model="gpt-5-mini"):
    prompt = build_rewrite_prompt(previous_turns, tool_turn, next_turn)
    client = _get_openai_client(client)
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=2000,
    )
    return response.output_text.strip()


ANNOTATION_PROMPT = dedent(
    """
You are an expert cognitive analyst. You will analyze a sequence of actions taken by two agents in a collaborative coding setting.

Your task is to classify **one target action** into a cognitive process category using the theoretical framework below.

---

## INPUT FORMAT

You are provided with:

Previous Action: <text describing what happened before the target action>
**Current Action**: <text describing the action to be labeled>
Next Action: <text describing what happens after the target action>

Broader Workflow Context:
- Previous Workflow Step: <workflow step before the matched step, if any>
- Matched Workflow Step: <matched workflow step>
- Next Workflow Step: <workflow step after the matched step, if any>

You must classify the **current_action** based on its function, using both the local action context and workflow context.

---

## OUTPUT FORMAT

Return a single JSON object with the following keys:

{
  "process_type": "planning | execution | feedback | control",
  "justification": "<brief reasoning for the process type, referencing the surrounding action or workflow context when relevant>"
}

---

## THEORETICAL FRAMEWORK

This scheme adapts the **Flower & Hayes (1981) Cognitive Process Model** for programming workflows.
Cognitive processes are recursive and context-dependent — planning, executing, and evaluating occur repeatedly throughout a coding session.

### 1. Process Categories

**PLANNING**
Formulating or revising goals, structuring code, or selecting solution strategies.
→ Examples: outlining function design, deciding on data structures, sketching pseudocode, reading task description before coding.

**EXECUTION**
Translating plans into concrete code or commands.
→ Examples: typing new functions, editing syntax, refactoring code, implementing logic, running code snippets.

**FEEDBACK / MONITORING**
Evaluating outputs, debugging, interpreting logs or test results, comparing alternatives.
→ Examples: reading error messages, analyzing stack traces, inspecting model output, consulting docs to validate reasoning.

**CONTROL / COORDINATION**
Meta-level actions that regulate task flow or transitions between phases.
→ Examples: pausing to reprioritize subtasks, switching from exploration to implementation, deferring a fix, or prompting the model for clarification.

---

## DECISION RULES

1. Use previous and next action context to interpret what cognitive function the current action serves.
2. Use the workflow context to disambiguate broad or underspecified actions.
3. Label actions by the function they perform, not by vague intention.
   - E.g., suggesting code = planning, executing code = execution, showing errors = feedback.

---

Previous Action: {previous_step_text}
Current Action: {current_step_text}
Next Action: {next_step_text}
Previous Workflow Step: {previous_workflow_step}
Matched Workflow Step: {matched_workflow_step}
Next Workflow Step: {next_workflow_step}
"""
)


OUTPUT_USE_NEXT_STEPS_PROMPT = dedent(
    """
You are an expert analyst studying how users integrate AI-generated content into their workflow.

Your task is to classify how the AI-assisted step at a specific moment is used in the subsequent workflow.

==================================================
INPUT DESCRIPTION
==================================================

You will receive the following inputs:

TASK_DESCRIPTION
- The overall task the user is trying to accomplish.
- This provides global context for understanding how the AI-assisted step might contribute.

PREVIOUS_HUMAN_STEP
- The nearest preceding human-authored step before the AI-assisted step being analyzed.
- This approximates what the user was doing or asking for around this moment.

CURRENT_AI_ASSISTED_STEP
- The current AI-assisted step being analyzed.
- This is the content whose downstream use you must evaluate.

WORKFLOW_CONTEXT
- The matched workflow step for the AI-assisted step, plus one workflow step before and one after.
- This is only secondary context and may be imperfect.
- Use it only as light background framing, not as primary evidence.

NEXT_STEPS
- A sequence of subsequent actions (from the user and/or tool) that occur after the AI output.
- These steps provide the primary behavioral evidence of how (or whether) the AI output was used.

Your classification must be based ONLY on observable evidence in NEXT_STEPS, while using TASK_DESCRIPTION and WORKFLOW_CONTEXT as contextual framing.
If NEXT_STEPS and WORKFLOW_CONTEXT seem to point in different directions, trust NEXT_STEPS.

==================================================
INPUT
==================================================

TASK_DESCRIPTION:
<<<
{task_description}
>>>

PREVIOUS_HUMAN_STEP:
<<<
{previous_human_step}
>>>

CURRENT_AI_ASSISTED_STEP:
<<<
{current_ai_assisted_step}
>>>

WORKFLOW_CONTEXT:
<<<
Previous Workflow Step: {previous_workflow_step}
Matched Workflow Step: {matched_workflow_step}
Next Workflow Step: {next_workflow_step}
>>>

NEXT_STEPS:
<<<
{next_steps_sequence}
>>>

==================================================

Your goal is to determine how the CURRENT_AI_ASSISTED_STEP is integrated into the workflow based on evidence in NEXT_STEPS.

--------------------------------------------------
CLASSIFICATION LABELS
--------------------------------------------------

Choose exactly ONE label.

- Reuse
The user directly reuses the AI-assisted content with minimal or trivial changes.
Examples:
- copying or pasting AI-generated text, code, or commands
- executing a command clearly provided by the AI
- accepting or applying an AI suggestion with only minor visible modification
- incorporating AI-generated wording or code into docs/files with little transformation

- Apply
The user takes an idea, method, structure, or recommendation from the AI-assisted content and adapts it to their own context in a visible way.
Examples:
- making a concrete edit, configuration change, test, or revision that clearly follows the AI's recommendation
- reimplementing an AI-suggested idea in a different form
- using AI recommendations as a basis for edits, tests, prompts, or plan changes that are clearly connected to that specific guidance but not direct copy-paste
- following the AI's suggested approach, but with noticeable adaptation rather than direct reuse

- Pushback
The user tests, questions, challenges, corrects, or follows up on the AI-assisted content because it appears insufficient, problematic, incomplete, or still unresolved.
Examples:
- identifying that the AI suggestion is wrong, incomplete, mismatched, or does not fully solve the problem
- asking follow-up questions, debugging, or testing because the AI output appears flawed or insufficient
- revising direction or seeking correction because the AI content was evaluated and found wanting

- Reject
There is no clear observable evidence that the AI-assisted content was meaningfully used in subsequent steps.
Examples:
- later steps are unrelated or only loosely related
- the user continues the task, but there is no visible trace that this specific AI-assisted step influenced those actions
- the connection is only thematic or speculative

--------------------------------------------------
DECISION GUIDANCE
--------------------------------------------------

Use a balanced standard:

- Do not require perfect proof for Apply. But Apply should only be used when NEXT_STEPS show a concrete downstream action that is specifically responsive to the AI-assisted content.
- Do not assign Reuse unless the connection is fairly direct.
- Prefer Reuse over Apply when the later steps show direct uptake of the AI content with little transformation, even if the user also verifies or lightly adjusts it afterward.
- Do not assign Reject merely because the later steps are not copy-paste explicit.
- Prefer Reject over Apply when the later steps are only generic continuation of the same task, such as browsing nearby files, generic testing, or continuing implementation without a clear visible trace of this specific AI-assisted step.
- Prefer Apply over Reject only when there is a specific visible connection between the AI-assisted content and what the user does next, such as implementing the recommended change, testing the suggested fix, or revising a file or plan in a way that clearly matches the guidance.
- Prefer Pushback when the user visibly responds to the AI output by probing, challenging, debugging, or seeking correction because the output seems wrong, incomplete, mismatched, or unresolved.
- Do not use Pushback for ordinary implementation or generic verification. Use it only when the later behavior indicates friction with the AI output itself.
- Prefer Reject when the relationship is only thematic, vague, or speculative.
- Prefer NEXT_STEPS over WORKFLOW_CONTEXT if they conflict.
- Similar topic alone is not enough; there should be some visible downstream trace.

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

1. Base your decision ONLY on observable evidence in NEXT_STEPS.
2. Do NOT infer hidden learning unless behaviorally reflected.
3. Do NOT use a staged or linear interpretation. Choose the single best-fitting label from the set above.
4. WORKFLOW_CONTEXT is secondary and may be imperfect.
5. Running tests, editing files, reading docs, or continuing the task support Apply only when they clearly instantiate or verify the AI's specific recommendation, not when they are merely generic continuation.
6. Return exactly one classification.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return a JSON object in this exact format:

{
  "label": "<Reuse | Apply | Pushback | Reject>",
  "justification": "<brief explanation referencing concrete evidence from NEXT_STEPS>"
}

Return ONLY the JSON object.
No additional commentary.
"""
)


def flatten_steps(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_steps: List[Dict[str, Any]] = []
    for item_index, item in enumerate(data_list):
        steps = item.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            flat_steps.append(
                {
                    "item_index": item_index,
                    "step_index": step_index,
                    "step": step,
                    "steps": steps,
                }
            )
    return flat_steps


def get_previous_human_step(steps: List[dict], current_step: dict) -> str:
    try:
        idx = steps.index(current_step)
    except ValueError:
        return "N/A"
    for j in range(idx - 1, -1, -1):
        step = steps[j]
        if step.get("actor") == "Human" and step.get("text"):
            return step.get("text", "")
    return "N/A"


def parse_json_object(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        parsed = json.loads(text.strip())
    except Exception:
        parsed = None
    return parsed if isinstance(parsed, dict) else fallback


def normalize_output_use_label(parsed: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    allowed_labels = {"Reuse", "Apply", "Pushback", "Reject"}
    label = parsed.get("label")
    justification = parsed.get("justification")

    if isinstance(label, str):
        normalized = label.strip()
        if normalized.lower().startswith("level"):
            parts = normalized.split("-", 1)
            normalized = parts[1].strip() if len(parts) == 2 else normalized
        if normalized in allowed_labels:
            return {
                "label": normalized,
                "justification": justification if isinstance(justification, str) else raw_text,
            }

    return {"label": None, "justification": raw_text}


def parse_json_list_or_wrap(text: str) -> List[Any]:
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    return parsed if isinstance(parsed, list) else [text]


def workflow_context(workflow: List[str], matched_index: int) -> Dict[str, Optional[str]]:
    previous_step = workflow[matched_index - 1] if matched_index - 1 >= 0 else None
    matched_step = workflow[matched_index] if 0 <= matched_index < len(workflow) else None
    next_step = workflow[matched_index + 1] if matched_index + 1 < len(workflow) else None
    return {
        "previous_workflow_step": previous_step,
        "matched_workflow_step": matched_step,
        "next_workflow_step": next_step,
    }


def annotate_output_use_next_steps(
    *,
    task_description: str,
    previous_human_step: str,
    current_ai_assisted_step: str,
    next_steps_sequence: List[str],
    workflow_ctx: Dict[str, Optional[str]],
    client: Optional[object] = None,
    model_name: str = "gpt-5-mini",
) -> Dict[str, Any]:
    prompt = OUTPUT_USE_NEXT_STEPS_PROMPT
    prompt = prompt.replace("{task_description}", task_description or "N/A")
    prompt = prompt.replace("{previous_human_step}", previous_human_step or "N/A")
    prompt = prompt.replace("{current_ai_assisted_step}", current_ai_assisted_step or "N/A")
    prompt = prompt.replace(
        "{previous_workflow_step}", workflow_ctx.get("previous_workflow_step") or "N/A"
    )
    prompt = prompt.replace(
        "{matched_workflow_step}", workflow_ctx.get("matched_workflow_step") or "N/A"
    )
    prompt = prompt.replace("{next_workflow_step}", workflow_ctx.get("next_workflow_step") or "N/A")
    prompt = prompt.replace("{next_steps_sequence}", "\n".join(next_steps_sequence) if next_steps_sequence else "N/A")

    client = _get_openai_client(client)
    completion = client.chat.completions.create(
        model=model_name,
        reasoning_effort="minimal",
        messages=[
            {
                "role": "system",
                "content": "You are a concise and consistent workflow integration annotation assistant.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    text = completion.choices[0].message.content.strip()
    parsed = parse_json_object(text, {"label": None, "justification": text})
    return normalize_output_use_label(parsed, text)


def annotate_step(
    current_step: Dict[str, Any],
    previous_step: Optional[Dict[str, Any]],
    next_step: Optional[Dict[str, Any]],
    workflow_ctx: Dict[str, Optional[str]],
    client: Optional[object] = None,
    model_name: str = "gpt-5-mini",
) -> Dict[str, Any]:
    def agent_name_prefix(step: Optional[Dict[str, Any]]) -> str:
        if not step:
            return ""
        return "AGENT 1: " if step.get("actor") == "Human" else "AGENT 2: "

    previous_step_text = (
        agent_name_prefix(previous_step) + previous_step.get("text", "") if previous_step else "N/A"
    )
    current_step_text = agent_name_prefix(current_step) + current_step.get("text", "")
    next_step_text = agent_name_prefix(next_step) + next_step.get("text", "") if next_step else "N/A"

    prompt = ANNOTATION_PROMPT
    prompt = prompt.replace("{previous_step_text}", previous_step_text or "N/A")
    prompt = prompt.replace("{current_step_text}", current_step_text or "N/A")
    prompt = prompt.replace("{next_step_text}", next_step_text or "N/A")
    prompt = prompt.replace(
        "{previous_workflow_step}", workflow_ctx.get("previous_workflow_step") or "N/A"
    )
    prompt = prompt.replace(
        "{matched_workflow_step}", workflow_ctx.get("matched_workflow_step") or "N/A"
    )
    prompt = prompt.replace("{next_workflow_step}", workflow_ctx.get("next_workflow_step") or "N/A")

    client = _get_openai_client(client)
    completion = client.chat.completions.create(
        model=model_name,
        reasoning_effort="minimal",
        messages=[
            {"role": "system", "content": "You are a concise and consistent cognitive annotation assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    text = completion.choices[0].message.content.strip()
    return parse_json_object(text, {"process_type": None, "justification": text})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label non-human turns in one *_annotated.json file using workflow context."
    )
    parser.add_argument(
        "--annotated-json",
        required=True,
        help="Path to input file ending with _annotated.json",
    )
    parser.add_argument(
        "--input-json",
        help="Optional raw merged JSON used only to compute time_taken from top-level time.start/time.end.",
    )
    parser.add_argument(
        "--time-split",
        required=True,
        choices=["long", "short"],
        help="Manual label indicating whether this run belongs to the long or short time split.",
    )
    parser.add_argument(
        "--workflow-txt",
        required=True,
        help="Path to workflow .txt file",
    )
    parser.add_argument(
        "--task-id",
        required=False,
        default="task",
        type=str,
        help="Free-form identifier for this task (recorded in the output JSON for bookkeeping). Default: 'task'.",
    )
    task_desc_group = parser.add_mutually_exclusive_group(required=True)
    task_desc_group.add_argument(
        "--task-description",
        type=str,
        help="Free-text description of the task being performed in the recording.",
    )
    task_desc_group.add_argument(
        "--task-description-file",
        type=str,
        help="Path to a text file containing the task description (alternative to --task-description).",
    )
    parser.add_argument(
        "--output-json",
        help="Optional explicit output path. Defaults to replacing _annotated.json with _annotated_labeled_workflow_context.json.",
    )
    parser.add_argument(
        "--output-use-only",
        action="store_true",
        help="Only compute workflow matching + output-use labels. Skip rewrite/counterfactual and cognitive annotations.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.annotated_json)
    raw_input_path = os.path.abspath(args.input_json) if args.input_json else None
    workflow_path = os.path.abspath(args.workflow_txt)

    if not os.path.isfile(input_path):
        print(f"Error: {input_path} is not a valid file.")
        return 1
    if not input_path.endswith("_annotated.json"):
        print("Error: input file must end with '_annotated.json'.")
        return 1
    if raw_input_path:
        if not os.path.isfile(raw_input_path):
            print(f"Error: {raw_input_path} is not a valid file.")
            return 1
        if not raw_input_path.endswith(".json"):
            print("Error: raw input file must end with '.json'.")
            return 1
    if not os.path.isfile(workflow_path):
        print(f"Error: {workflow_path} is not a valid file.")
        return 1
    if not workflow_path.endswith(".txt"):
        print("Error: workflow file must end with '.txt'.")
        return 1

    file_index = get_idx_from_filepath(input_path)

    data_list: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = [line.strip() for line in f if line.strip()]

    merged_data: Optional[Any] = None
    if raw_input_path:
        with open(raw_input_path, "r", encoding="utf-8") as f:
            merged_data = json.load(f)

    time_taken = extract_time_taken(raw_input_path)
    flat_steps = flatten_steps(data_list)
    total_steps = len(flat_steps)
    non_human_counts = 0
    matching_workflow_indices: List[int] = []
    rewrite_map: Dict[int, List[Any]] = {}
    cognitive_annotations: List[Dict[str, Any]] = []
    output_use_annotations: List[Dict[str, Any]] = []
    if args.task_description_file:
        task_desc_path = os.path.abspath(args.task_description_file)
        if not os.path.isfile(task_desc_path):
            print(f"Error: {task_desc_path} is not a valid file.")
            return 1
        with open(task_desc_path, "r", encoding="utf-8") as f:
            task_description_text = f.read().strip()
    else:
        task_description_text = args.task_description.strip()
    if not task_description_text:
        print("Error: task description is empty.")
        return 1
    task_description_dict = {
        "task_id": args.task_id,
        "task_description": task_description_text,
    }
    previous_matched_index: Optional[int] = None

    for flat_idx, entry in enumerate(flat_steps):
        step = entry["step"]
        if step.get("actor") == "Human":
            continue

        non_human_counts += 1
        previous_step = flat_steps[flat_idx - 1]["step"] if flat_idx > 0 else None
        next_step = flat_steps[flat_idx + 1]["step"] if flat_idx + 1 < len(flat_steps) else None
        previous_human_step = get_previous_human_step(entry["steps"], step)
        merged_hints: List[str] = []
        if should_use_merged_hints(step.get("text", "")):
            merged_hints = top_merged_hints(
                merged_data,
                current_ai_assisted_step=step.get("text", ""),
                previous_human_step=previous_human_step,
                previous_step=previous_step,
                next_step=next_step,
            )
        matched_index = llm_match_step_to_workflow(
            current_ai_assisted_step=step.get("text", ""),
            workflow=workflow,
            previous_human_step=previous_human_step,
            previous_step=previous_step,
            next_step=next_step,
            previous_matched_index=previous_matched_index,
            merged_hints=merged_hints,
            model="gpt-5-mini",
        )
        matching_workflow_indices.append(matched_index)
        previous_matched_index = matched_index
        workflow_ctx = workflow_context(workflow, matched_index)

        if not args.output_use_only:
            previous_workflow_steps = workflow[max(0, matched_index - 3):matched_index]
            next_workflow_step = workflow[matched_index + 1] if matched_index + 1 < len(workflow) else None
            rewritten_steps = rewrite_tool_turn(
                previous_turns=previous_workflow_steps,
                tool_turn=step,
                next_turn=next_workflow_step,
                model="gpt-5-mini",
            )
            rewrite_map[matched_index] = parse_json_list_or_wrap(rewritten_steps)

            labelled_ai_step = annotate_step(
                current_step=step,
                previous_step=previous_step,
                next_step=next_step,
                workflow_ctx=workflow_ctx,
            )
            cognitive_annotations.append(
                {
                    "label": labelled_ai_step,
                    "step": step,
                    "workflow_context": workflow_ctx,
                    "item_index": entry["item_index"],
                    "step_index": entry["step_index"],
                }
            )

        next_steps_sequence = workflow[matched_index + 1: matched_index + 6]
        output_use_label = annotate_output_use_next_steps(
            task_description=task_description_dict["task_description"],
            previous_human_step=previous_human_step,
            current_ai_assisted_step=step.get("text", ""),
            next_steps_sequence=next_steps_sequence,
            workflow_ctx=workflow_ctx,
            model_name="gpt-5-mini",
        )
        output_use_annotations.append(
            {
                "label": output_use_label,
                "step": step,
                "matched_workflow_index": matched_index,
                "workflow_context": workflow_ctx,
                "next_steps_sequence": next_steps_sequence,
                "item_index": entry["item_index"],
                "step_index": entry["step_index"],
            }
        )

    workflow_indices_set = set(matching_workflow_indices)
    workflow_coverage = len(workflow_indices_set) / len(workflow) if workflow else 0

    labeled_output = {
        "filename": input_path,
        "input_json": raw_input_path,
        "index": file_index,
        "data": data_list,
        "rewrite_map": rewrite_map,
        "ai_fraction_workflow": workflow_coverage,
        "cognitive_annotations": cognitive_annotations,
        "output_use_annotations": output_use_annotations,
        "task_description": task_description_dict,
        "non_human_counts": non_human_counts,
        "matched_workflow_indices": matching_workflow_indices,
        "total_steps": total_steps,
        "time_taken": time_taken,
        "time_split": args.time_split,
        "workflow": workflow,
        "workflow_context_window": 1,
        "labeling_script": os.path.basename(__file__),
        "output_use_only": args.output_use_only,
    }

    if args.output_use_only:
        default_output_path = input_path.replace(
            "_annotated.json", "_annotated_output_use_workflow_context.json"
        )
    else:
        default_output_path = input_path.replace(
            "_annotated.json", "_annotated_labeled_workflow_context.json"
        )
    output_path = os.path.abspath(args.output_json) if args.output_json else default_output_path
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(labeled_output, out_f)
    print(f"Saved labeled output to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
