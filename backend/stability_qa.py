"""Camera stability QA with full-clip probe plus capped refinement."""

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .sampler import get_video_duration
from .utils import clip_artifacts_dir, format_timestamp

log = logging.getLogger("pov_qa.stability_qa")

STABILITY_REPORT_SCHEMA_VERSION = 4


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _seed_for_clip(filename: str) -> int:
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) ^ int(config.RANDOM_SEED)) & 0xFFFFFFFF


def _estimate_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Estimate global camera motion magnitude between two gray frames."""
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=config.STABILITY_MAX_FEATURES,
        qualityLevel=config.STABILITY_FEATURE_QUALITY,
        minDistance=10,
    )
    if corners is None or len(corners) < 4:
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, corners, None,
        winSize=(21, 21), maxLevel=3,
    )
    if next_pts is None:
        return 0.0

    good_mask = status.flatten() == 1
    if np.sum(good_mask) < 4:
        return 0.0

    src = corners[good_mask].reshape(-1, 2)
    dst = next_pts[good_mask].reshape(-1, 2)

    disp = dst - src
    disp_mag = np.linalg.norm(disp, axis=1)
    median_disp = float(np.median(disp_mag))
    p90_disp = float(np.percentile(disp_mag, 90))

    transform, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if transform is None:
        return p90_disp

    tx = transform[0, 2]
    ty = transform[1, 2]
    translation = np.sqrt(tx ** 2 + ty ** 2)

    cos_a = transform[0, 0]
    sin_a = transform[1, 0]
    angle_rad = abs(np.arctan2(sin_a, cos_a))
    h = prev_gray.shape[0]
    rotation_equiv = angle_rad * h / 2

    model_motion = float(translation + rotation_equiv)
    capped_model = min(model_motion, p90_disp * 2.5)
    return float(0.7 * median_disp + 0.3 * capped_model)


def _resize_if_needed(frame: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _stability_config_signature() -> dict:
    return {
        "schema_version": STABILITY_REPORT_SCHEMA_VERSION,
        "spike_threshold": float(config.STABILITY_SPIKE_THRESHOLD),
        "min_fail_segment_sec": float(config.STABILITY_MIN_FAIL_SEGMENT_SEC),
        "max_analysis_sec": float(config.STABILITY_MAX_ANALYSIS_SEC),
        "probe_coverage": float(config.STABILITY_PROBE_COVERAGE),
        "probe_window_sec": float(config.STABILITY_PROBE_WINDOW_SEC),
        "probe_k_min": int(config.STABILITY_PROBE_K_MIN),
        "probe_k_max": int(config.STABILITY_PROBE_K_MAX),
        "probe_fps": float(config.STABILITY_PROBE_FPS),
        "probe_max_dim": int(config.STABILITY_PROBE_MAX_DIM),
        "probe_jitter_fraction": float(config.STABILITY_PROBE_JITTER_FRACTION),
        "refine_top_fraction": float(config.STABILITY_REFINE_TOP_FRACTION),
        "refine_min_windows": int(config.STABILITY_REFINE_MIN_WINDOWS),
        "refine_max_windows": int(config.STABILITY_REFINE_MAX_WINDOWS),
        "refine_coverage_fraction": float(config.STABILITY_REFINE_COVERAGE_FRACTION),
        "refine_fps": float(config.STABILITY_REFINE_FPS),
        "refine_max_dim": int(config.STABILITY_REFINE_MAX_DIM),
    }


def _window_to_dict(start: float, end: float, source: str | None = None,
                    score: float | None = None) -> dict:
    out = {
        "start_time": round(float(start), 2),
        "end_time": round(float(end), 2),
        "start_fmt": format_timestamp(float(start)),
        "end_fmt": format_timestamp(float(end)),
    }
    if source is not None:
        out["source"] = source
    if score is not None:
        out["score"] = round(float(score), 2)
    return out


def _merge_windows(windows: list[tuple[float, float]], limit: float) -> list[tuple[float, float]]:
    cleaned = []
    for start, end in windows:
        s = max(0.0, float(start))
        e = min(float(limit), float(end))
        if e <= s:
            continue
        cleaned.append((s, e))
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: x[0])

    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_probe_windows(duration: float, clip_name: str) -> list[tuple[float, float]]:
    if duration <= 0:
        return []

    window_sec = float(config.STABILITY_PROBE_WINDOW_SEC)
    if duration <= window_sec:
        return [(0.0, duration)]

    coverage = float(config.STABILITY_PROBE_COVERAGE)
    k_raw = math.ceil((duration * coverage) / max(window_sec, 1e-6))
    k = int(_clamp(k_raw, int(config.STABILITY_PROBE_K_MIN), int(config.STABILITY_PROBE_K_MAX)))

    half = window_sec / 2.0
    centers = np.linspace(half, duration - half, k)

    jitter_amp = float(config.STABILITY_PROBE_JITTER_FRACTION) * window_sec
    rng = np.random.default_rng(_seed_for_clip(clip_name))
    jittered = []
    for c in centers:
        jc = float(c + rng.uniform(-jitter_amp, jitter_amp))
        jc = float(np.clip(jc, half, duration - half))
        jittered.append(jc)

    windows = [(c - half, c + half) for c in jittered]
    return _merge_windows(windows, limit=duration)


def _collect_probe_jitter_from_windows(video_path: Path,
                                       windows: list[tuple[float, float]],
                                       sample_fps: float,
                                       max_dim: int) -> tuple[list[float], list[float], list[dict]]:
    """Cheap probe metric: phase-correlation translation jitter."""
    if not windows:
        return [], [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], []

    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if video_fps <= 0:
            video_fps = 30.0
        step = max(1, int(round(video_fps / max(sample_fps, 1e-6))))

        jitter_scores: list[float] = []
        timestamps: list[float] = []
        window_scores: list[dict] = []

        for start, end in windows:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(start)) * 1000.0)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            prev_gray = None
            prev_ts = None
            prev_shift = None
            stream_ended = False
            local_scores: list[float] = []

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    stream_ended = True
                    break

                ts = frame_idx / video_fps
                frame_idx += 1

                if ts < start:
                    continue
                if ts > end:
                    break

                frame = _resize_if_needed(frame, max_dim)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

                if prev_gray is not None and prev_ts is not None:
                    shift, _ = cv2.phaseCorrelate(prev_gray, gray)
                    dx, dy = float(shift[0]), float(shift[1])
                    if prev_shift is not None:
                        jx = dx - prev_shift[0]
                        jy = dy - prev_shift[1]
                        jitter = float(np.hypot(jx, jy))
                        mid_ts = (float(prev_ts) + float(ts)) / 2.0
                        jitter_scores.append(jitter)
                        timestamps.append(mid_ts)
                        local_scores.append(jitter)
                    prev_shift = (dx, dy)

                prev_gray = gray
                prev_ts = ts

                for _ in range(step - 1):
                    if not cap.grab():
                        stream_ended = True
                        break
                    frame_idx += 1
                if stream_ended:
                    break

            agg = float(np.percentile(np.array(local_scores, dtype=np.float32), 90)) if local_scores else 0.0
            window_scores.append(_window_to_dict(start, end, source="probe", score=agg))

            if stream_ended:
                break

        return jitter_scores, timestamps, window_scores
    finally:
        cap.release()


def _collect_motion_from_windows(video_path: Path,
                                 windows: list[tuple[float, float]],
                                 sample_fps: float,
                                 max_dim: int = 640) -> tuple[list[float], list[float]]:
    """Heavy refine: LK+affine only on selected windows."""
    if not windows:
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []

    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if video_fps <= 0:
            video_fps = 30.0
        step = max(1, int(round(video_fps / max(sample_fps, 1e-6))))

        motion_scores: list[float] = []
        timestamps: list[float] = []

        for start, end in windows:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(start)) * 1000.0)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            prev_gray = None
            prev_ts = None
            stream_ended = False

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    stream_ended = True
                    break

                ts = frame_idx / video_fps
                frame_idx += 1

                if ts < start:
                    continue
                if ts > end:
                    break

                frame = _resize_if_needed(frame, max_dim)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None and prev_ts is not None:
                    mag = _estimate_motion(prev_gray, curr_gray)
                    motion_scores.append(float(mag))
                    timestamps.append((float(prev_ts) + float(ts)) / 2.0)

                prev_gray = curr_gray
                prev_ts = ts

                for _ in range(step - 1):
                    if not cap.grab():
                        stream_ended = True
                        break
                    frame_idx += 1
                if stream_ended:
                    break

            if stream_ended:
                break

        return motion_scores, timestamps
    finally:
        cap.release()


def _compute_spike_segments(motion_scores: list[float],
                            timestamps: list[float],
                            sample_fps: float) -> list[dict]:
    if not motion_scores or not timestamps:
        return []

    window_sec = float(config.STABILITY_WINDOW_SEC)
    threshold = float(config.STABILITY_SPIKE_THRESHOLD)
    # Bucket by absolute clip time (1s windows by default). This avoids false
    # long segments when we analyze disjoint probe/refine windows.
    bucket_scores: dict[int, list[float]] = {}
    for ts, score in zip(timestamps, motion_scores):
        bucket_id = int(float(ts) // max(window_sec, 1e-6))
        bucket_scores.setdefault(bucket_id, []).append(float(score))

    hot_buckets: list[tuple[int, float]] = []
    for bucket_id, values in bucket_scores.items():
        if not values:
            continue
        bucket_score = float(np.percentile(np.array(values, dtype=np.float32), 90))
        if bucket_score > threshold:
            hot_buckets.append((bucket_id, bucket_score))

    if not hot_buckets:
        return []

    hot_buckets.sort(key=lambda x: x[0])

    spike_segments = []
    seg_start_bucket = hot_buckets[0][0]
    seg_end_bucket = hot_buckets[0][0]
    seg_peak = hot_buckets[0][1]

    for bucket_id, bucket_score in hot_buckets[1:]:
        if bucket_id == seg_end_bucket + 1:
            seg_end_bucket = bucket_id
            seg_peak = max(seg_peak, bucket_score)
            continue

        seg_start = seg_start_bucket * window_sec
        seg_end = (seg_end_bucket + 1) * window_sec
        spike_segments.append({
            "start_time": round(seg_start, 2),
            "end_time": round(seg_end, 2),
            "start_fmt": format_timestamp(seg_start),
            "end_fmt": format_timestamp(seg_end),
            "motion_score": round(seg_peak, 2),
        })
        seg_start_bucket = bucket_id
        seg_end_bucket = bucket_id
        seg_peak = bucket_score

    seg_start = seg_start_bucket * window_sec
    seg_end = (seg_end_bucket + 1) * window_sec
    spike_segments.append({
        "start_time": round(seg_start, 2),
        "end_time": round(seg_end, 2),
        "start_fmt": format_timestamp(seg_start),
        "end_fmt": format_timestamp(seg_end),
        "motion_score": round(seg_peak, 2),
    })

    return spike_segments


def _select_refine_windows(probe_window_scores: list[dict],
                           probe_spike_segments: list[dict],
                           duration: float) -> list[tuple[float, float]]:
    if not probe_window_scores:
        return []

    threshold = float(config.STABILITY_SPIKE_THRESHOLD)
    top_fraction = float(config.STABILITY_REFINE_TOP_FRACTION)
    window_sec = float(config.STABILITY_PROBE_WINDOW_SEC)

    ranked = sorted(probe_window_scores, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    top_n = max(1, int(math.ceil(len(ranked) * top_fraction)))
    top_set = ranked[:top_n]

    by_key: dict[tuple[float, float], dict] = {}

    for item in ranked:
        if float(item.get("score", 0.0)) >= threshold:
            key = (float(item["start_time"]), float(item["end_time"]))
            by_key[key] = item

    for item in top_set:
        key = (float(item["start_time"]), float(item["end_time"]))
        by_key[key] = item

    # Ensure probe spikes get local refinement windows.
    pad = float(config.STABILITY_CANDIDATE_PAD_SEC)
    for seg in probe_spike_segments:
        s = float(seg["start_time"]) - pad
        e = float(seg["end_time"]) + pad
        key = (round(max(0.0, s), 2), round(min(duration, e), 2))
        by_key[key] = {
            "start_time": key[0],
            "end_time": key[1],
            "score": float(seg.get("motion_score", 0.0)),
        }

    candidates = sorted(by_key.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if not candidates:
        return []

    count_cap = int(_clamp(
        math.ceil(top_fraction * len(probe_window_scores)),
        int(config.STABILITY_REFINE_MIN_WINDOWS),
        int(config.STABILITY_REFINE_MAX_WINDOWS),
    ))
    coverage_cap = max(
        1,
        int(math.floor((duration * float(config.STABILITY_REFINE_COVERAGE_FRACTION)) / max(window_sec, 1e-6))),
    )
    cap = min(len(candidates), count_cap, coverage_cap)

    selected = candidates[:cap]
    pairs = [(float(x["start_time"]), float(x["end_time"])) for x in selected]
    return _merge_windows(pairs, limit=duration)


def _sum_segment_duration(segments: list[dict]) -> float:
    return float(sum(max(0.0, float(s["end_time"]) - float(s["start_time"])) for s in segments))


def run_stability_qa(video_path: Path,
                     artifacts_dir: Path | None = None) -> dict:
    """Analyze camera stability for a single clip."""
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    out_dir = clip_artifacts_dir(video_path.name, artifacts_dir)
    report_path = out_dir / "stability_report.json"

    expected_sig = _stability_config_signature()
    if config.USE_CACHED_REPORTS and report_path.exists():
        try:
            with open(report_path) as f:
                cached = json.load(f)
            if (
                int(cached.get("num_frame_pairs", -1)) >= 0
                and cached.get("stability_config") == expected_sig
            ):
                log.info("Stability QA cache hit: %s", video_path.name)
                return cached
        except Exception:
            pass

    log.info("Stability QA: %s", video_path.name)
    total_t0 = time.perf_counter()

    duration = float(get_video_duration(video_path))
    max_analysis_sec = float(config.STABILITY_MAX_ANALYSIS_SEC)
    analysis_limit = min(duration, max_analysis_sec) if duration > 0 else 0.0
    analysis_truncated = bool(duration > max_analysis_sec)

    probe_windows = _build_probe_windows(analysis_limit, video_path.name)
    if not probe_windows and analysis_limit > 0:
        probe_windows = [(0.0, analysis_limit)]

    probe_t0 = time.perf_counter()
    probe_scores, probe_timestamps, probe_window_scores = _collect_probe_jitter_from_windows(
        video_path,
        probe_windows,
        sample_fps=float(config.STABILITY_PROBE_FPS),
        max_dim=int(config.STABILITY_PROBE_MAX_DIM),
    )
    probe_t1 = time.perf_counter()

    if not probe_scores:
        timing_sec = {
            "probe": round(probe_t1 - probe_t0, 3),
            "refine": 0.0,
            "total": round(time.perf_counter() - total_t0, 3),
        }
        report = {
            "filename": video_path.name,
            "max_motion_score": 0.0,
            "peak_motion_score": 0.0,
            "mean_motion_score": 0.0,
            "median_motion_score": 0.0,
            "spike_segments": [],
            "probe_spike_segments": [],
            "total_shaky_sec": 0.0,
            "num_frame_pairs": 0,
            "sample_fps": float(config.STABILITY_PROBE_FPS),
            "window_sec": float(config.STABILITY_WINDOW_SEC),
            "spike_threshold": float(config.STABILITY_SPIKE_THRESHOLD),
            "min_fail_segment_sec": float(config.STABILITY_MIN_FAIL_SEGMENT_SEC),
            "analysis_truncated": analysis_truncated,
            "clip_duration_sec": round(duration, 2),
            "analyzed_duration_sec": round(analysis_limit, 2),
            "coarse_candidate_windows": [_window_to_dict(s, e, source="probe") for s, e in probe_windows],
            "refined_windows_analyzed": [],
            "probe_window_scores": probe_window_scores,
            "coverage_ratio": round(
                (sum(max(0.0, e - s) for s, e in probe_windows) / analysis_limit) if analysis_limit > 0 else 0.0,
                4,
            ),
            "timing_sec": timing_sec,
            "requires_manual_confirmation": True,
            "spike_source": "probe",
            "stability_config": expected_sig,
            "stability_schema_version": STABILITY_REPORT_SCHEMA_VERSION,
            "error": "insufficient sampled frames",
        }
        _save_report(report, out_dir)
        return report

    probe_arr = np.array(probe_scores, dtype=np.float32)
    probe_spike_segments = _compute_spike_segments(
        probe_scores,
        probe_timestamps,
        float(config.STABILITY_PROBE_FPS),
    )

    refine_pairs = _select_refine_windows(probe_window_scores, probe_spike_segments, analysis_limit)
    refine_t0 = time.perf_counter()
    refine_scores, refine_timestamps = _collect_motion_from_windows(
        video_path,
        windows=refine_pairs,
        sample_fps=float(config.STABILITY_REFINE_FPS),
        max_dim=int(config.STABILITY_REFINE_MAX_DIM),
    )
    refine_t1 = time.perf_counter()

    if refine_scores:
        refine_arr = np.array(refine_scores, dtype=np.float32)
        final_spikes = _compute_spike_segments(
            refine_scores,
            refine_timestamps,
            float(config.STABILITY_REFINE_FPS),
        )
        max_motion = round(float(np.percentile(refine_arr, 99)), 2)
        peak_motion = round(float(np.max(refine_arr)), 2)
        mean_motion = round(float(np.mean(refine_arr)), 2)
        median_motion = round(float(np.median(refine_arr)), 2)
        num_pairs = len(refine_scores)
        sample_fps = float(config.STABILITY_REFINE_FPS)
        spike_source = "refine"
    else:
        final_spikes = probe_spike_segments
        max_motion = round(float(np.percentile(probe_arr, 99)), 2)
        peak_motion = round(float(np.max(probe_arr)), 2)
        mean_motion = round(float(np.mean(probe_arr)), 2)
        median_motion = round(float(np.median(probe_arr)), 2)
        num_pairs = len(probe_scores)
        sample_fps = float(config.STABILITY_PROBE_FPS)
        spike_source = "probe"

    requires_manual_confirmation = bool(
        spike_source == "refine"
        and probe_spike_segments
        and not final_spikes
    )

    timing_sec = {
        "probe": round(probe_t1 - probe_t0, 3),
        "refine": round(refine_t1 - refine_t0, 3),
        "total": round(time.perf_counter() - total_t0, 3),
    }

    probe_coverage = (
        sum(max(0.0, float(e) - float(s)) for s, e in probe_windows) / analysis_limit
        if analysis_limit > 0 else 0.0
    )

    report = {
        "filename": video_path.name,
        "max_motion_score": max_motion,
        "peak_motion_score": peak_motion,
        "mean_motion_score": mean_motion,
        "median_motion_score": median_motion,
        "spike_segments": final_spikes,
        "probe_spike_segments": probe_spike_segments,
        "total_shaky_sec": round(_sum_segment_duration(final_spikes), 2),
        "probe_total_shaky_sec": round(_sum_segment_duration(probe_spike_segments), 2),
        "num_frame_pairs": num_pairs,
        "sample_fps": round(sample_fps, 3),
        "probe_sample_fps": float(config.STABILITY_PROBE_FPS),
        "window_sec": float(config.STABILITY_WINDOW_SEC),
        "spike_threshold": float(config.STABILITY_SPIKE_THRESHOLD),
        "min_fail_segment_sec": float(config.STABILITY_MIN_FAIL_SEGMENT_SEC),
        "analysis_truncated": analysis_truncated,
        "clip_duration_sec": round(duration, 2),
        "analyzed_duration_sec": round(analysis_limit, 2),
        "coarse_candidate_windows": [_window_to_dict(s, e, source="probe") for s, e in probe_windows],
        "refined_windows_analyzed": [_window_to_dict(s, e, source="refine") for s, e in refine_pairs],
        "probe_window_scores": probe_window_scores,
        "coverage_ratio": round(float(probe_coverage), 4),
        "timing_sec": timing_sec,
        "requires_manual_confirmation": requires_manual_confirmation,
        "spike_source": spike_source,
        "stability_config": expected_sig,
        "stability_schema_version": STABILITY_REPORT_SCHEMA_VERSION,
    }

    _save_report(report, out_dir)
    return report


def _save_report(report: dict, out_dir: Path) -> None:
    with open(out_dir / "stability_report.json", "w") as f:
        json.dump(report, f, indent=2)


def check_c9(report: dict) -> tuple[bool, str, str]:
    """Evaluate stability from the stability report.

    Returns (passed, detail_string, gate_status).
    """
    max_motion = float(report.get("max_motion_score", 0.0))
    spikes = report.get("spike_segments", [])
    analysis_truncated = bool(report.get("analysis_truncated", False))
    analyzed_duration_sec = float(report.get("analyzed_duration_sec", 0.0))
    min_fail_sec = float(report.get("min_fail_segment_sec", config.STABILITY_MIN_FAIL_SEGMENT_SEC))

    qualifying = [
        s for s in spikes
        if (float(s["end_time"]) - float(s["start_time"])) > min_fail_sec
    ]
    if qualifying:
        worst = max(qualifying, key=lambda s: float(s.get("motion_score", 0.0)))
        seg_desc = ", ".join(f"{s['start_fmt']}-{s['end_fmt']}" for s in qualifying[:3])
        detail = (
            f"super shaky segment around {seg_desc} "
            f"(motion_score={float(worst.get('motion_score', 0.0)):.2f})"
        )
        return False, detail, "fail"

    # Early severe shake should still hard-fail even if a burst is short.
    early_spike = None
    early_window_sec = float(config.STABILITY_EARLY_SPIKE_WINDOW_SEC)
    early_score_fail = float(config.STABILITY_EARLY_SPIKE_SCORE_FAIL)
    for s in spikes:
        if (float(s.get("start_time", 0.0)) <= early_window_sec
                and float(s.get("motion_score", 0.0)) >= early_score_fail):
            early_spike = s
            break
    if early_spike is not None:
        detail = (
            f"severe early shake around "
            f"{early_spike['start_fmt']}-{early_spike['end_fmt']} "
            f"(motion_score={float(early_spike.get('motion_score', 0.0)):.2f})"
        )
        return False, detail, "fail"

    if bool(report.get("requires_manual_confirmation", False)):
        detail = "probe jitter seen but not confirmed by refine; treated as stable"
        return True, detail, "pass"

    if analysis_truncated:
        clip_duration_sec = float(report.get("clip_duration_sec", analyzed_duration_sec))
        detail = (
            f"partial scan only ({analyzed_duration_sec:.0f}s/{clip_duration_sec:.0f}s); "
            "manual stability confirmation required"
        )
        return False, detail, "borderline"

    if spikes:
        longest = max((float(s["end_time"]) - float(s["start_time"]) for s in spikes), default=0.0)
        seg_desc = ", ".join(f"{s['start_fmt']}-{s['end_fmt']}" for s in spikes[:3])
        detail = (
            f"brief shaky bursts accepted "
            f"(longest={longest:.1f}s <= {min_fail_sec:.1f}s) around {seg_desc}"
        )
        return True, detail, "pass"

    return True, f"stable (max_motion={max_motion:.1f})", "pass"
