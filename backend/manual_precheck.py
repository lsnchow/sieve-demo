"""Heuristic quality prechecks to speed up manual review."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from . import config
from .sampler import iter_sampled_frames
from .utils import clip_artifacts_dir, format_timestamp

log = logging.getLogger("pov_qa.manual_precheck")


def _histogram(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (np.sum(hist) + 1e-6)
    return hist.astype(np.float32)


def _find_segments(timestamps: list[float], flags: list[bool]) -> list[dict]:
    segs = []
    in_seg = False
    start = 0.0
    for ts, f in zip(timestamps, flags):
        if f and not in_seg:
            in_seg = True
            start = ts
        elif (not f) and in_seg:
            in_seg = False
            segs.append({
                "start_time": round(start, 2),
                "end_time": round(ts, 2),
                "start_fmt": format_timestamp(start),
                "end_fmt": format_timestamp(ts),
            })
    if in_seg and timestamps:
        end = timestamps[-1]
        segs.append({
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "start_fmt": format_timestamp(start),
            "end_fmt": format_timestamp(end),
        })
    return segs


def run_manual_precheck(video_path: Path,
                        artifacts_dir: Path | None = None) -> dict:
    """Generate heuristic flags for manual QA speed-up."""
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    out_dir = clip_artifacts_dir(video_path.name, artifacts_dir)
    report_path = out_dir / "manual_precheck.json"

    if config.USE_CACHED_REPORTS and report_path.exists():
        try:
            with open(report_path) as f:
                return json.load(f)
        except Exception:
            pass

    timestamps = []
    lumas = []
    blurs = []
    cut_scores = []
    cut_candidates = []
    motion_diffs = []
    duplicate_flags = []

    prev_hist = None
    prev_ts = None
    prev_small = None
    prev_hash = None
    frame_count = 0

    for ts, frame in iter_sampled_frames(
        video_path,
        sample_fps=config.MANUAL_PRECHECK_SAMPLE_FPS,
        max_dim=640,
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        frame_hash = small > float(np.mean(small))

        luma = float(np.mean(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        curr_hist = _histogram(gray)

        cut_score = 0.0
        if prev_hist is not None:
            cut_score = float(cv2.compareHist(
                prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA
            ))
            # Candidate jump cut when distribution changes abruptly.
            if cut_score >= config.MANUAL_SCENE_CUT_THRESHOLD:
                cut_candidates.append({
                    "time": round(ts, 2),
                    "time_fmt": format_timestamp(ts),
                    "score": round(cut_score, 3),
                    "prev_time": round(prev_ts if prev_ts is not None else ts, 2),
                })

        if prev_small is not None:
            diff = float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))) / 255.0)
            motion_diffs.append(diff)
        if prev_hash is not None:
            hamming_ratio = float(np.mean(frame_hash != prev_hash))
            duplicate_flags.append(hamming_ratio <= config.MANUAL_DUPLICATE_HAMMING_RATIO_MAX)

        timestamps.append(float(ts))
        lumas.append(luma)
        blurs.append(blur)
        cut_scores.append(cut_score)
        prev_hist = curr_hist
        prev_ts = float(ts)
        prev_small = small
        prev_hash = frame_hash
        frame_count += 1

    if frame_count == 0:
        report = {
            "filename": video_path.name,
            "sample_fps": config.MANUAL_PRECHECK_SAMPLE_FPS,
            "frames_sampled": 0,
            "error": "no frames extracted",
            "candidate_jump_cuts": [],
            "dark_segments": [],
            "blurry_segments": [],
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # Sort strongest cut candidates and limit output for reviewer readability.
    cut_candidates = sorted(cut_candidates, key=lambda x: x["score"], reverse=True)
    cut_candidates = cut_candidates[:config.MANUAL_MAX_CUT_CANDIDATES]

    dark_flags = [l < config.MANUAL_DARK_LUMA_THRESHOLD for l in lumas]
    blur_flags = [b < config.MANUAL_BLUR_LAPLACIAN_THRESHOLD for b in blurs]
    static_flags = [d < config.MANUAL_STATIC_DIFF_THRESHOLD for d in motion_diffs]

    dark_segments = _find_segments(timestamps, dark_flags)
    blurry_segments = _find_segments(timestamps, blur_flags)
    static_ratio = round(float(np.mean(static_flags)) if static_flags else 0.0, 4)
    duplicate_ratio = round(float(np.mean(duplicate_flags)) if duplicate_flags else 0.0, 4)
    motion_score = round(float(np.mean(motion_diffs)) if motion_diffs else 0.0, 4)
    static_duplicate_score = round(float(min(1.0, (0.7 * static_ratio) + (0.3 * duplicate_ratio))), 4)

    report = {
        "filename": video_path.name,
        "sample_fps": config.MANUAL_PRECHECK_SAMPLE_FPS,
        "frames_sampled": frame_count,
        "avg_luma": round(float(np.mean(lumas)), 2),
        "dark_ratio": round(float(np.mean(dark_flags)), 4),
        "avg_blur_laplacian_var": round(float(np.mean(blurs)), 2),
        "blurry_ratio": round(float(np.mean(blur_flags)), 4),
        "motion_score": motion_score,
        "static_ratio": static_ratio,
        "duplicate_ratio": duplicate_ratio,
        "static_duplicate_score": static_duplicate_score,
        "max_cut_score": round(float(np.max(cut_scores) if cut_scores else 0.0), 3),
        "candidate_jump_cuts": cut_candidates,
        "dark_segments": dark_segments,
        "blurry_segments": blurry_segments,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
