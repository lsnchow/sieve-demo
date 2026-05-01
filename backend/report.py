"""Dataset-curation export surface: manifest, index, thumbnails, and summary."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from pathlib import Path

import cv2
import pandas as pd

from . import config
from .curation import encode_json, encode_list, split_semicolon_values

log = logging.getLogger("pov_qa.report")


def export_live_snapshot(artifacts_dir: Path | None = None,
                         output_dir: Path | None = None,
                         expected_total: int | None = None) -> dict:
    """Rebuild manifest/index exports incrementally during review/labeling."""
    summary = export_deliverables(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        expected_total=expected_total,
    )
    if summary:
        log.info(
            "Live snapshot updated: recommended=%s needs_review=%s low_value=%s total=%s",
            summary.get(config.PUBLIC_STATUS_RECOMMENDED, 0),
            summary.get(config.PUBLIC_STATUS_NEEDS_REVIEW, 0),
            summary.get(config.PUBLIC_STATUS_LOW_VALUE, 0),
            summary.get("total_clips", 0),
        )
    return summary


def export_deliverables(artifacts_dir: Path | None = None,
                        output_dir: Path | None = None,
                        expected_total: int | None = None) -> dict:
    """Generate dataset manifest, JSON export, SQLite index, thumbnails, and summary."""
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    output_dir = output_dir or config.EXPORT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = output_dir / "clips"
    thumbs_dir = output_dir / "thumbnails"
    clips_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    qa_path = artifacts_dir / "qa_metrics.csv"
    if not qa_path.exists():
        log.error("qa_metrics.csv not found — run the automated pipeline first")
        return {}

    qa_df = pd.read_csv(qa_path)
    manifest_rows = []

    for _, row in qa_df.iterrows():
        manifest_rows.append(_build_manifest_row(row, artifacts_dir, clips_dir, thumbs_dir))

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["status", "filename"]).reset_index(drop=True)

    manifest_csv_path = output_dir / "dataset_manifest.csv"
    manifest_json_path = output_dir / "dataset_manifest.json"
    summary_path = output_dir / "summary.json"
    sqlite_path = output_dir / "dataset_index.sqlite"

    manifest_df.to_csv(manifest_csv_path, index=False)
    manifest_json = json.loads(manifest_df.to_json(orient="records"))
    manifest_json_path.write_text(json.dumps(manifest_json, indent=2))

    summary = _build_summary(manifest_df, output_dir)
    if expected_total is not None and len(manifest_df) != expected_total:
        summary["expected_total"] = expected_total
        summary["expected_total_match"] = False
    else:
        summary["expected_total_match"] = True if expected_total is not None else None
    summary_path.write_text(json.dumps(summary, indent=2))

    _write_sqlite_index(manifest_df, sqlite_path)

    if bool(config.WRITE_RATIONALE_REPORTS):
        _export_rationale_markdown(manifest_df, output_dir)

    print(f"\n{'=' * 52}")
    print("Dataset Curation Export")
    print(f"  Total clips:    {summary['total_clips']}")
    print(f"  Recommended:    {summary[config.PUBLIC_STATUS_RECOMMENDED]}")
    print(f"  Needs review:   {summary[config.PUBLIC_STATUS_NEEDS_REVIEW]}")
    print(f"  Low value:      {summary[config.PUBLIC_STATUS_LOW_VALUE]}")
    print(f"{'=' * 52}")

    return summary


def _build_manifest_row(row: pd.Series,
                        artifacts_dir: Path,
                        clips_dir: Path,
                        thumbs_dir: Path) -> dict:
    raw = row.to_dict()
    filename = raw["filename"]
    stem = Path(filename).stem
    clip_dir = artifacts_dir / stem

    manual_review = _load_json(clip_dir / "manual_review.json")
    labels = _load_json(clip_dir / "labels.json")
    hands = _load_json(clip_dir / "hands_report.json")
    precheck = _load_json(clip_dir / "manual_precheck.json")
    stability = _load_json(clip_dir / "stability_report.json")
    wide = _load_json(clip_dir / "wide_lens_report.json")

    quality_flags = split_semicolon_values(raw.get("quality_flags", ""))
    curation_reasons = split_semicolon_values(raw.get("curation_reasons", ""))

    manual_status = "pending"
    if manual_review:
        if manual_review.get("approved", False):
            manual_status = "recommended"
        else:
            manual_status = "low_value"
            curation_reasons.extend(split_semicolon_values(manual_review.get("reject_reasons", [])))

    label_status = "done" if labels else "pending"

    status = str(raw.get("curation_status", config.PUBLIC_STATUS_NEEDS_REVIEW))
    if raw.get("gate_result") == config.GATE_FAIL:
        status = config.PUBLIC_STATUS_LOW_VALUE
    elif manual_review:
        status = config.PUBLIC_STATUS_RECOMMENDED if manual_review.get("approved", False) else config.PUBLIC_STATUS_LOW_VALUE
    else:
        status = config.PUBLIC_STATUS_NEEDS_REVIEW
        curation_reasons.append("manual_review_pending")

    thumbnail_path, thumbnail_token = _copy_thumbnail(
        clip_dir,
        thumbs_dir,
        stem,
        raw.get("source_path", ""),
    )
    clip_export_path = _copy_clip_if_selected(raw.get("source_path", ""), clips_dir, status)

    if precheck and float(precheck.get("static_duplicate_score", 0.0)) >= config.STATIC_DUPLICATE_FLAG_SCORE:
        quality_flags.append("static_or_duplicate")

    manifest_row = {
        "filename": filename,
        "status": status,
        "training_value_score": raw.get("training_value_score"),
        "quality_flags": encode_list(quality_flags),
        "curation_reasons": encode_list(curation_reasons),
        "demo_category": raw.get("demo_category", ""),
        "source_title": raw.get("source_title", ""),
        "source_url": raw.get("source_url", ""),
        "license": raw.get("license", ""),
        "attribution": raw.get("attribution", ""),
        "clip_start_sec": raw.get("clip_start_sec", ""),
        "clip_duration_sec": raw.get("clip_duration_sec", ""),
        "source_path": raw.get("source_path", ""),
        "exported_clip_path": clip_export_path,
        "thumbnail_path": thumbnail_path,
        "thumbnail_token": thumbnail_token,
        "display_w": raw.get("display_w"),
        "display_h": raw.get("display_h"),
        "effective_fps": raw.get("effective_fps"),
        "duration": raw.get("duration"),
        "brightness_score": raw.get("brightness_score"),
        "blur_score": raw.get("blur_score"),
        "motion_score": raw.get("motion_score"),
        "camera_stability_score": raw.get("camera_stability_score"),
        "static_duplicate_score": raw.get("static_duplicate_score"),
        "dark_ratio": raw.get("dark_ratio"),
        "blurry_ratio": raw.get("blurry_ratio"),
        "max_cut_score": raw.get("max_cut_score"),
        "hand_visible_ratio": raw.get("hand_visible_ratio", raw.get("c3a_hands_ratio")),
        "hand_presence_score": raw.get("hand_presence_score"),
        "hand_motion_score": raw.get("hand_motion_score"),
        "hand_count_distribution": raw.get("hand_count_distribution", encode_json({})),
        "hand_region_distribution": raw.get("hand_region_distribution", encode_json({})),
        "egocentric_proxy_score": raw.get("egocentric_proxy_score"),
        "wide_lens_score": raw.get("c3b_wide_score"),
        "wide_lens_confidence": wide.get("confidence", ""),
        "max_motion_score": raw.get("c9_max_motion"),
        "total_shaky_sec": stability.get("total_shaky_sec", ""),
        "manual_status": manual_status,
        "label_status": label_status,
        "task": labels.get("task", ""),
        "objects": labels.get("objects", ""),
        "task_outcome": labels.get("task_outcome", ""),
        "environment": labels.get("environment", ""),
        "time_of_day": labels.get("time_of_day", ""),
        "notes": labels.get("notes", ""),
        "gate_result": raw.get("gate_result", ""),
        "hands_detail": raw.get("c3a_detail", ""),
        "perspective_detail": raw.get("c3b_detail", ""),
        "stability_detail": raw.get("c9_detail", ""),
        "missing_hand_segments": encode_json(hands.get("missing_segments", [])),
        "dark_segments": encode_json(precheck.get("dark_segments", [])),
        "blurry_segments": encode_json(precheck.get("blurry_segments", [])),
        "jump_cut_candidates": encode_json(precheck.get("candidate_jump_cuts", [])),
    }
    return manifest_row


def _copy_thumbnail(clip_dir: Path,
                    thumbs_dir: Path,
                    stem: str,
                    source_path: str = "") -> tuple[str, str]:
    keyframes_dir = clip_dir / "keyframes"
    candidates = sorted(keyframes_dir.glob("*.jpg"))
    out_path = thumbs_dir / f"{stem}.jpg"

    if candidates:
        selected = _select_representative_keyframe(candidates)
        shutil.copy2(selected, out_path)
    else:
        if not source_path:
            return "", ""
        source = Path(source_path)
        if not source.exists():
            return "", ""
        if not _extract_middle_frame_thumbnail(source, out_path):
            return "", ""

    return f"thumbnails/{out_path.name}", str(out_path.stat().st_mtime_ns)


def _select_representative_keyframe(candidates: list[Path]) -> Path:
    """Pick a thumbnail that avoids black intro frames and favors central, visible content."""
    if len(candidates) == 1:
        return candidates[0]

    start = max(0, len(candidates) // 5)
    end = min(len(candidates), len(candidates) - start)
    window = candidates[start:end] or candidates

    best_path = window[len(window) // 2]
    best_score = float("-inf")
    center_index = (len(candidates) - 1) / 2.0

    for path in window:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        mean_luma = float(image.mean()) / 255.0
        contrast = float(image.std()) / 128.0
        normalized_index = candidates.index(path)
        center_penalty = abs(normalized_index - center_index) / max(center_index, 1.0)

        score = (0.65 * mean_luma) + (0.35 * contrast) - (0.15 * center_penalty)
        if score > best_score:
            best_score = score
            best_path = path

    return best_path


def _extract_middle_frame_thumbnail(video_path: Path, out_path: Path) -> bool:
    """Extract a representative thumbnail from the middle of the source video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return False

        middle_frame = max(frame_count // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False

        return bool(cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]))
    finally:
        cap.release()


def _copy_clip_if_selected(source_path: str, clips_dir: Path, status: str) -> str:
    if status not in set(config.CLIP_EXPORT_STATUSES):
        return ""
    if not source_path:
        return ""
    source = Path(source_path)
    if not source.exists():
        return ""
    out_path = clips_dir / source.name
    if not out_path.exists() or out_path.stat().st_size != source.stat().st_size:
        shutil.copy2(source, out_path)
    return f"clips/{out_path.name}"


def _build_summary(manifest_df: pd.DataFrame, output_dir: Path) -> dict:
    counts = manifest_df["status"].value_counts().to_dict()
    category_counts = manifest_df["demo_category"].fillna("").value_counts().to_dict()

    quality_flag_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for value in manifest_df["quality_flags"].fillna(""):
        for flag in split_semicolon_values(value):
            quality_flag_counts[flag] = quality_flag_counts.get(flag, 0) + 1
    for value in manifest_df["curation_reasons"].fillna(""):
        for reason in split_semicolon_values(value):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "total_clips": int(len(manifest_df)),
        config.PUBLIC_STATUS_RECOMMENDED: int(counts.get(config.PUBLIC_STATUS_RECOMMENDED, 0)),
        config.PUBLIC_STATUS_NEEDS_REVIEW: int(counts.get(config.PUBLIC_STATUS_NEEDS_REVIEW, 0)),
        config.PUBLIC_STATUS_LOW_VALUE: int(counts.get(config.PUBLIC_STATUS_LOW_VALUE, 0)),
        "demo_category_counts": category_counts,
        "quality_flag_counts": quality_flag_counts,
        "curation_reason_counts": reason_counts,
        "clip_export_dir": str(output_dir / "clips"),
        "thumbnail_dir": str(output_dir / "thumbnails"),
    }


def _write_sqlite_index(manifest_df: pd.DataFrame, sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        manifest_df.to_sql("clips", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON clips(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_demo_category ON clips(demo_category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_training_value_score ON clips(training_value_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_flags ON clips(quality_flags)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_curation_reasons ON clips(curation_reasons)")
        conn.commit()


def _export_rationale_markdown(manifest_df: pd.DataFrame, output_dir: Path) -> None:
    rationale_dir = output_dir / "rationale"
    rationale_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest_df.iterrows():
        lines = [
            f"# Clip Summary — {row['filename']}",
            "",
            f"- Status: `{row['status']}`",
            f"- Training value score: `{row['training_value_score']}`",
            f"- Demo category: `{row['demo_category']}`",
            f"- Quality flags: {row['quality_flags'] or '[none]'}",
            f"- Curation reasons: {row['curation_reasons'] or '[none]'}",
            f"- Hand visible ratio: {row['hand_visible_ratio']}",
            f"- Hand presence score: {row['hand_presence_score']}",
            f"- Hand motion score: {row['hand_motion_score']}",
            f"- Egocentric proxy score: {row['egocentric_proxy_score']}",
            f"- Brightness score: {row['brightness_score']}",
            f"- Blur score: {row['blur_score']}",
            f"- Motion score: {row['motion_score']}",
            f"- Camera stability score: {row['camera_stability_score']}",
            f"- Static/duplicate score: {row['static_duplicate_score']}",
            "",
        ]
        (rationale_dir / f"{Path(row['filename']).stem}.md").write_text("\n".join(lines))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}
