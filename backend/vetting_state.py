"""Run-level vetting state persistence for resumable sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .reviewer import build_vetting_dashboard

VETTING_STATE_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_vetting_state(artifacts_dir: Path | None = None) -> dict:
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    state_path = artifacts_dir / "vetting_state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception:
        return {}


def write_vetting_state(state: dict,
                        artifacts_dir: Path | None = None) -> Path:
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifacts_dir / "vetting_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    return state_path


def build_vetting_state(artifacts_dir: Path | None = None,
                        output_dir: Path | None = None,
                        qa_metrics_path: Path | None = None,
                        last_clip: str | None = None,
                        started_at: str | None = None) -> dict:
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    output_dir = output_dir or config.EXPORT_DIR
    qa_metrics_path = qa_metrics_path or (artifacts_dir / "qa_metrics.csv")

    dashboard = build_vetting_dashboard(
        artifacts_dir=artifacts_dir,
        qa_metrics_path=qa_metrics_path if qa_metrics_path.exists() else None,
    )

    existing = load_vetting_state(artifacts_dir)
    if started_at is None:
        started_at = existing.get("started_at", _now_iso())

    if dashboard.empty:
        reviewed_count = 0
        labeled_count = 0
        pending_review: list[str] = []
        pending_label: list[str] = []
    else:
        reviewed_count = int((dashboard["manual_status"] != "pending").sum())
        labeled_count = int((dashboard["label_status"] == "done").sum())
        pending_review = dashboard.loc[
            dashboard["next_step"] == "review", "filename"
        ].tolist()
        pending_label = dashboard.loc[
            dashboard["next_step"] == "label", "filename"
        ].tolist()

    state = {
        "version": VETTING_STATE_VERSION,
        "qa_metrics_path": str(qa_metrics_path),
        "output_dir": str(output_dir),
        "started_at": started_at,
        "updated_at": _now_iso(),
        "reviewed_count": reviewed_count,
        "labeled_count": labeled_count,
        "pending_review": pending_review,
        "pending_label": pending_label,
        "last_clip": last_clip or existing.get("last_clip", ""),
    }
    return state


def update_and_write_vetting_state(artifacts_dir: Path | None = None,
                                   output_dir: Path | None = None,
                                   qa_metrics_path: Path | None = None,
                                   last_clip: str | None = None) -> Path:
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    state = build_vetting_state(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        qa_metrics_path=qa_metrics_path,
        last_clip=last_clip,
    )
    return write_vetting_state(state, artifacts_dir=artifacts_dir)
