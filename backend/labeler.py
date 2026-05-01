"""Labeling for recommended clips: interactive CLI with controlled vocabulary."""

import json
import logging
from pathlib import Path
from typing import Callable

from . import config
from .prompting import get_prompter
from .utils import clip_artifacts_dir

log = logging.getLogger("pov_qa.labeler")
PROMPTER = get_prompter()


def _prompt_choice(prompt: str, choices: list[str], allow_custom: bool = False) -> str:
    """Prompt user to select from a list of choices."""
    options = [(c, c) for c in choices]
    if allow_custom:
        options.append(("[custom]", "__custom__"))
    value = PROMPTER.select(f"{prompt}", options, default=choices[0] if choices else None)
    if value == "__custom__":
        return PROMPTER.text("Enter custom value", allow_empty=False)
    return value


def _prompt_text(prompt: str, required: bool = True) -> str:
    """Prompt for free-text input."""
    return PROMPTER.text(prompt, allow_empty=not required)


def _collect_label_fields(filename: str, clip_dir: Path) -> dict:
    """Collect label fields for a single clip."""
    print(f"\n{'='*60}")
    print(f"Labeling: {filename}")
    print(f"{'='*60}")
    print("Label spec:")
    print("  - task: [verb] [object] [context] (specific)")
    print("  - objects: comma-separated interacted objects")
    print("  - task_outcome: success | fail | partial")
    print("  - environment: specific location type")
    print("  - time_of_day: daytime | nighttime | indoor")
    print("  - notes: optional")

    summary_path = clip_dir / "review_summary.txt"
    if summary_path.exists():
        print(f"Review summary: {summary_path}")
    sheet_path = clip_dir / "review_sheet.png"
    if sheet_path.exists():
        print(f"Contact sheet:  {sheet_path}")

    task = _prompt_text(
        "Task description ([verb] [object] [context], e.g. 'washing dishes in sink')",
        required=True,
    )
    objects = _prompt_text(
        "Objects interacted with (comma-separated, e.g. 'plate, sponge, faucet')",
        required=True,
    )
    task_outcome = _prompt_choice("Task outcome:", config.TASK_OUTCOMES)
    environment = _prompt_choice(
        "Environment:", config.ENVIRONMENTS, allow_custom=True
    )
    time_of_day = _prompt_choice("Time of day:", config.TIMES_OF_DAY)
    notes = _prompt_text("Notes (optional)", required=False)

    return {
        "filename": filename,
        "task": task,
        "objects": objects,
        "task_outcome": task_outcome,
        "environment": environment,
        "time_of_day": time_of_day,
        "notes": notes,
    }


def _save_label(label: dict, artifacts_dir: Path) -> Path:
    clip_dir = clip_artifacts_dir(label["filename"], artifacts_dir)
    label_path = clip_dir / "labels.json"
    with open(label_path, "w") as f:
        json.dump(label, f, indent=2)
    log.info("Labels saved: %s", label_path)
    return label_path


def prompt_label_for_clip(filename: str,
                          artifacts_dir: Path | None = None,
                          allow_skip: bool = True,
                          allow_quit: bool = True,
                          on_label_saved: Callable[[dict], bool] | None = None) -> dict:
    """Prompt action for one clip and optionally save labels.

    Returns {"action": "saved"|"skip"|"quit", "label": dict|None}
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    clip_dir = clip_artifacts_dir(filename, artifacts_dir)

    options = [("Label now", "label")]
    if allow_skip:
        options.append(("Skip for now", "skip"))
    if allow_quit:
        options.append(("Quit labeling session", "quit"))

    action = PROMPTER.select(
        f"{filename}: action",
        options,
        default="label",
    )
    if action == "skip":
        return {"action": "skip", "label": None}
    if action == "quit":
        return {"action": "quit", "label": None}

    label = _collect_label_fields(filename, clip_dir)
    _save_label(label, artifacts_dir=artifacts_dir)
    if on_label_saved is not None:
        should_continue = on_label_saved(label)
        if should_continue is False:
            return {"action": "quit", "label": label}
    return {"action": "saved", "label": label}


def label_clip(filename: str,
               artifacts_dir: Path | None = None,
               on_label_saved: Callable[[dict], bool] | None = None) -> dict:
    """Interactively label a single recommended clip and save it.

    Returns label dict.
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    clip_dir = clip_artifacts_dir(filename, artifacts_dir)
    label = _collect_label_fields(filename, clip_dir)
    _save_label(label, artifacts_dir=artifacts_dir)
    if on_label_saved is not None:
        on_label_saved(label)
    return label


def interactive_labeling(artifacts_dir: Path | None = None,
                         on_label_saved: Callable[[dict], bool] | None = None) -> list[dict]:
    """Label all recommended clips interactively.

    Reads manual_review.json files to determine which clips are recommended.
    Returns list of label dicts.
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR

    if not artifacts_dir.exists():
        print("No artifacts directory found — run autogate and review first.")
        return []

    approved = []
    for d in sorted(artifacts_dir.iterdir()):
        if not d.is_dir():
            continue
        review_path = d / "manual_review.json"
        if review_path.exists():
            with open(review_path) as f:
                review = json.load(f)
            if review.get("approved"):
                approved.append(review["filename"])

    if not approved:
        print("No recommended clips found — run manual review first.")
        return []

    print(f"\n{len(approved)} recommended clips to label.")

    labels = []
    existing = _load_existing_labels(artifacts_dir)

    for i, filename in enumerate(approved):
        stem = Path(filename).stem
        if stem in existing:
            print(f"\n[{i+1}/{len(approved)}] {filename} — already labeled, skipping")
            labels.append(existing[stem])
            continue

        print(f"\n[{i+1}/{len(approved)}]")
        result = prompt_label_for_clip(
            filename,
            artifacts_dir=artifacts_dir,
            allow_skip=True,
            allow_quit=True,
            on_label_saved=on_label_saved,
        )
        if result["action"] == "skip":
            continue
        if result["action"] == "quit":
            break
        if result["label"] is not None:
            labels.append(result["label"])

    return labels


def _load_existing_labels(artifacts_dir: Path) -> dict:
    """Load previously completed labels."""
    existing = {}
    for d in artifacts_dir.iterdir():
        if d.is_dir():
            label_file = d / "labels.json"
            if label_file.exists():
                with open(label_file) as f:
                    label = json.load(f)
                existing[d.name] = label
    return existing
