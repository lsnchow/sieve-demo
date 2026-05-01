"""Gate aggregation: combine automated checks, classify PASS/FAIL/BORDERLINE."""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config
from .curation import (
    compute_curation_reasons,
    compute_quality_flags,
    compute_training_value_score,
    encode_json,
    encode_list,
    public_status_from_row,
)
from .hands_qa import check_c3a, run_hands_qa
from .ingest import check_c1, check_c2, run_ingest
from .manual_precheck import run_manual_precheck
from .sampler import extract_keyframes
from .stability_qa import check_c9, run_stability_qa
from .utils import collect_video_files
from .wide_lens_qa import check_c3b, run_wide_lens_qa

log = logging.getLogger("pov_qa.gate")


def _summarize_precheck(precheck_report: dict | None) -> dict:
    report = precheck_report or {}
    return {
        "dark_ratio": report.get("dark_ratio"),
        "blurry_ratio": report.get("blurry_ratio"),
        "motion_score": report.get("motion_score"),
        "static_ratio": report.get("static_ratio"),
        "duplicate_ratio": report.get("duplicate_ratio"),
        "static_duplicate_score": report.get("static_duplicate_score"),
        "max_cut_score": report.get("max_cut_score"),
    }


def _augment_curation_fields(row: dict,
                             precheck_report: dict | None = None,
                             hands_report: dict | None = None) -> dict:
    row.update(_summarize_precheck(precheck_report))

    if hands_report:
        row["hand_visible_ratio"] = hands_report.get("hand_visible_ratio")
        row["hand_presence_score"] = hands_report.get("hand_presence_score")
        row["hand_motion_score"] = hands_report.get("hand_motion_score")
        row["hand_count_distribution"] = encode_json(
            hands_report.get("hand_count_distribution", {})
        )
        row["hand_region_distribution"] = encode_json(
            hands_report.get("hand_region_distribution", {})
        )
        row["egocentric_proxy_score"] = hands_report.get("egocentric_proxy_score")
    else:
        row["hand_visible_ratio"] = row.get("c3a_hands_ratio")
        row["hand_presence_score"] = None
        row["hand_motion_score"] = None
        row["hand_count_distribution"] = encode_json({})
        row["hand_region_distribution"] = encode_json({})
        row["egocentric_proxy_score"] = None

    score, components = compute_training_value_score(row, precheck_report=precheck_report)
    flags = compute_quality_flags(row, precheck_report=precheck_report)
    reasons = compute_curation_reasons(row, flags)
    status = public_status_from_row(row, score, flags)

    row["training_value_score"] = score
    row["quality_flags"] = encode_list(flags)
    row["curation_reasons"] = encode_list(reasons)
    row["curation_status"] = status

    row["brightness_score"] = (
        round(components["brightness_score"], 4)
        if components["brightness_score"] is not None else None
    )
    row["blur_score"] = (
        round(components["blur_score"], 4)
        if components["blur_score"] is not None else None
    )
    row["motion_score"] = (
        round(components["motion_score"], 4)
        if components["motion_score"] is not None else None
    )
    row["camera_stability_score"] = (
        round(components["camera_stability_score"], 4)
        if components["camera_stability_score"] is not None else None
    )
    return row


def run_autogate(input_dir: Path,
                 expected_count: int | None = None,
                 artifacts_dir: Path | None = None) -> pd.DataFrame:
    """Run full automated gate on all clips.

    1. Ingest and format checks
    2. Stability QA
    3. Perspective QA
    4. Hands QA
    5. Classify each clip

    Returns DataFrame with per-clip gate results.
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timing_enabled = bool(config.ENABLE_STAGE_TIMING)

    # Phase A: Ingest
    log.info("=" * 60)
    log.info("Phase A: Ingest & Metadata")
    log.info("=" * 60)
    metadata_list = run_ingest(input_dir, expected_count, artifacts_dir)

    files = collect_video_files(input_dir)
    file_map = {f.name: f for f in files}

    rows = []
    for meta in tqdm(metadata_list, desc="Running automated gate"):
        clip_t0 = time.perf_counter() if timing_enabled else None
        t_widelens_sec = None
        t_stability_sec = None
        t_hands_sec = None
        precheck_report = None

        filename = meta["filename"]
        video_path = file_map[filename]

        c1_pass, c1_detail = check_c1(meta)
        c2_pass, c2_detail = check_c2(meta)
        precheck_report = run_manual_precheck(video_path, artifacts_dir=artifacts_dir)

        reject_reasons = []
        if not c1_pass:
            reject_reasons.append(c1_detail)
        if not c2_pass:
            reject_reasons.append(c2_detail)

        # Extract keyframes for review regardless
        log.info("Extracting keyframes: %s", filename)
        extract_keyframes(video_path,
                          output_dir=artifacts_dir / Path(filename).stem / "keyframes")

        # Phase B: Vision-based checks
        # For the local demo UI, always collect the diagnostic signals even when
        # a clip already fails a format gate. This keeps the artifact surface
        # informative instead of half-empty.
        hands_report = None
        stab_report = None
        c3a_pass, c3a_detail, c3a_gate = False, "skipped", "skip"
        c9_pass, c9_detail, c9_gate = False, "skipped", "skip"
        log.info("Running stability QA: %s", filename)
        stab_t0 = time.perf_counter() if timing_enabled else None
        stab_report = run_stability_qa(video_path, artifacts_dir)
        if timing_enabled:
            t_stability_sec = time.perf_counter() - stab_t0
        c9_pass, c9_detail, c9_gate = check_c9(stab_report)
        c9_hard_fail = (not c9_pass and c9_gate == "fail")
        if c9_hard_fail:
            reject_reasons.append(c9_detail)

        log.info("Running wide-lens QA: %s", filename)
        wl_t0 = time.perf_counter() if timing_enabled else None
        wl_report = run_wide_lens_qa(video_path, artifacts_dir)
        if timing_enabled:
            t_widelens_sec = time.perf_counter() - wl_t0
        c3b_pass, c3b_detail, c3b_gate = check_c3b(wl_report)
        c3b_hard_fail = (not c3b_pass and c3b_gate == "fail")
        if c3b_hard_fail:
            reject_reasons.append(c3b_detail)

        log.info("Running hands QA: %s", filename)
        hands_t0 = time.perf_counter() if timing_enabled else None
        hands_report = run_hands_qa(video_path, artifacts_dir)
        if timing_enabled:
            t_hands_sec = time.perf_counter() - hands_t0
        c3a_pass, c3a_detail, c3a_gate = check_c3a(hands_report)
        if not c3a_pass and c3a_gate == "fail":
            reject_reasons.append(c3a_detail)

        # Gate decision
        gate_result = _classify_gate(
            c1_pass, c2_pass, c3a_pass, c3a_gate,
            c3b_pass, c3b_gate, c9_pass, c9_gate,
        )

        row = _build_row(
            meta, c1_pass, c1_detail, c2_pass, c2_detail,
            hands_ratio=hands_report["hands_ratio"] if hands_report else None,
            c3a_pass=c3a_pass, c3a_detail=c3a_detail, c3a_gate=c3a_gate,
            wide_score=wl_report["wide_lens_score"],
            c3b_pass=c3b_pass, c3b_detail=c3b_detail, c3b_gate=c3b_gate,
            max_motion=stab_report["max_motion_score"] if stab_report else None,
            c9_pass=c9_pass, c9_detail=c9_detail, c9_gate=c9_gate,
            gate_result=gate_result,
            reject_reasons=reject_reasons,
            t_widelens_sec=t_widelens_sec,
            t_stability_sec=t_stability_sec,
            t_hands_sec=t_hands_sec,
            t_total_clip_sec=(time.perf_counter() - clip_t0) if timing_enabled else None,
        )
        row = _augment_curation_fields(
            row,
            precheck_report=precheck_report,
            hands_report=hands_report,
        )
        rows.append(row)

        log.info(
            "%s => %s | resolution=%s fps=%s hands=%s perspective=%s stability=%s",
            filename, gate_result,
            "PASS" if c1_pass else "FAIL",
            "PASS" if c2_pass else "FAIL",
            c3a_gate.upper(),
            c3b_gate.upper(),
            c9_gate.upper(),
        )

    df = pd.DataFrame(rows)

    csv_path = artifacts_dir / "qa_metrics.csv"
    df.to_csv(csv_path, index=False)
    log.info("Gate results saved to %s", csv_path)

    # Summary
    counts = df["gate_result"].value_counts()
    log.info("Gate summary:")
    for result, count in counts.items():
        log.info("  %s: %d", result, count)

    if config.WRITE_TIMING_REPORT and timing_enabled:
        _write_timing_report(df, artifacts_dir)

    return df


def _classify_gate(c1_pass, c2_pass, c3a_pass, c3a_gate,
                   c3b_pass, c3b_gate, c9_pass, c9_gate) -> str:
    """Determine overall gate result."""
    if not c1_pass or not c2_pass:
        return config.GATE_FAIL

    hard_fails = []
    borderlines = []

    for passed, gate_status in [
        (c3a_pass, c3a_gate),
        (c3b_pass, c3b_gate),
        (c9_pass, c9_gate),
    ]:
        if not passed:
            if gate_status == "fail":
                hard_fails.append(True)
            elif gate_status == "borderline":
                borderlines.append(True)

    if hard_fails:
        return config.GATE_FAIL
    if borderlines:
        return config.GATE_BORDERLINE
    return config.GATE_PASS


def _build_row(meta, c1_pass, c1_detail, c2_pass, c2_detail,
               hands_ratio, c3a_pass, c3a_detail, c3a_gate,
               wide_score, c3b_pass, c3b_detail, c3b_gate,
               max_motion, c9_pass, c9_detail, c9_gate,
               gate_result, reject_reasons,
               t_widelens_sec=None, t_stability_sec=None,
               t_hands_sec=None, t_total_clip_sec=None) -> dict:
    return {
        "filename": meta["filename"],
        "source_path": meta.get("source_path", ""),
        "source_url": meta.get("source_url", ""),
        "license": meta.get("license", ""),
        "attribution": meta.get("attribution", ""),
        "demo_category": meta.get("demo_category", ""),
        "source_title": meta.get("source_title", ""),
        "clip_start_sec": meta.get("clip_start_sec", ""),
        "clip_duration_sec": meta.get("clip_duration_sec", ""),
        "display_w": meta["display_w"],
        "display_h": meta["display_h"],
        "effective_fps": meta["effective_fps"],
        "duration": meta["duration"],
        "c1_pass": c1_pass,
        "c1_detail": c1_detail,
        "c2_pass": c2_pass,
        "c2_detail": c2_detail,
        "c3a_hands_ratio": hands_ratio,
        "c3a_pass": c3a_pass,
        "c3a_detail": c3a_detail,
        "c3a_gate": c3a_gate,
        "c3b_wide_score": wide_score,
        "c3b_pass": c3b_pass,
        "c3b_detail": c3b_detail,
        "c3b_gate": c3b_gate,
        "c9_max_motion": max_motion,
        "c9_pass": c9_pass,
        "c9_detail": c9_detail,
        "c9_gate": c9_gate,
        "gate_result": gate_result,
        "reject_reasons": "; ".join(reject_reasons) if reject_reasons else "",
        "t_widelens_sec": round(float(t_widelens_sec), 3) if t_widelens_sec is not None else None,
        "t_stability_sec": round(float(t_stability_sec), 3) if t_stability_sec is not None else None,
        "t_hands_sec": round(float(t_hands_sec), 3) if t_hands_sec is not None else None,
        "t_total_clip_sec": round(float(t_total_clip_sec), 3) if t_total_clip_sec is not None else None,
    }


def _write_timing_report(df: pd.DataFrame, artifacts_dir: Path) -> None:
    cols = ["t_widelens_sec", "t_stability_sec", "t_hands_sec", "t_total_clip_sec"]
    if any(c not in df.columns for c in cols):
        return

    timing_df = df[cols].fillna(0.0)
    summary = {
        "clips": int(len(df)),
        "sum_sec": {c: round(float(timing_df[c].sum()), 3) for c in cols},
        "mean_sec": {c: round(float(timing_df[c].mean()), 3) for c in cols},
        "p95_sec": {c: round(float(timing_df[c].quantile(0.95)), 3) for c in cols},
    }

    out = artifacts_dir / "timing_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Timing summary saved to %s", out)
