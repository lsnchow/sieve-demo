"""C9 tuning helpers: stability-only sweep and threshold A/B analysis."""

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config
from .stability_qa import run_stability_qa
from .utils import collect_video_files

log = logging.getLogger("pov_qa.c9_tuning")


def _list_video_files(input_dir: Path) -> list[Path]:
    return collect_video_files(input_dir)


def _load_report(report_path: Path) -> dict:
    with open(report_path) as f:
        return json.load(f)


def build_c9_metrics(input_dir: Path,
                     artifacts_dir: Path | None = None,
                     force: bool = False) -> pd.DataFrame:
    """Run/collect C9 reports and return per-clip metrics dataframe."""
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    files = _list_video_files(input_dir)
    rows = []

    for video in tqdm(files, desc="C9 stability sweep"):
        clip_dir = artifacts_dir / video.stem
        report_path = clip_dir / "stability_report.json"

        if force or not report_path.exists():
            report = run_stability_qa(video, artifacts_dir=artifacts_dir)
        else:
            report = _load_report(report_path)

        segments = report.get("spike_segments", [])
        durations = [max(0.0, s["end_time"] - s["start_time"]) for s in segments]
        report_total_shaky_sec = float(
            report.get("total_shaky_sec", sum(durations))
        )

        rows.append({
            "filename": report["filename"],
            "max_motion_score": float(report.get("max_motion_score", 0.0)),
            "peak_motion_score": float(
                report.get("peak_motion_score", report.get("max_motion_score", 0.0))
            ),
            "mean_motion_score": float(report.get("mean_motion_score", 0.0)),
            "median_motion_score": float(report.get("median_motion_score", 0.0)),
            "spike_segments": int(len(segments)),
            "longest_spike_sec": round(max(durations) if durations else 0.0, 2),
            "total_spike_sec": round(sum(durations), 2),
            "report_total_shaky_sec": round(report_total_shaky_sec, 2),
            "analysis_truncated": bool(report.get("analysis_truncated", False)),
            "early_terminated": bool(report.get("early_terminated", False)),
            "num_frame_pairs": int(report.get("num_frame_pairs", 0)),
        })

    df = pd.DataFrame(rows).sort_values("max_motion_score", ascending=False)
    return df


def build_threshold_sweep(metrics_df: pd.DataFrame,
                          thresholds: list[float],
                          min_longest_spike_sec: float = 0.0,
                          min_total_shaky_sec: float = 0.0,
                          score_col: str = "max_motion_score") -> pd.DataFrame:
    """Build A/B table for reject decisions over thresholds.

    Reject rule used for tuning:
    score_col > threshold AND
    (
      longest_spike_sec >= min_longest_spike_sec OR
      report_total_shaky_sec >= min_total_shaky_sec
    )
    """
    rows = []
    for t in thresholds:
        longest_mask = metrics_df["longest_spike_sec"] >= min_longest_spike_sec
        total_shaky_mask = metrics_df["report_total_shaky_sec"] >= min_total_shaky_sec
        duration_mask = longest_mask | total_shaky_mask
        mask = (
            (metrics_df[score_col] > t)
            & duration_mask
        )
        rejects = metrics_df.loc[mask, "filename"].tolist()
        rows.append({
            "threshold": t,
            "score_col": score_col,
            "min_longest_spike_sec": min_longest_spike_sec,
            "min_total_shaky_sec": min_total_shaky_sec,
            "rejected_count": len(rejects),
            "rejected_files": "; ".join(rejects),
        })
    return pd.DataFrame(rows)
