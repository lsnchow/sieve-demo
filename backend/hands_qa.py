"""C3a: Hands visible QA — MediaPipe Hands + reliability layer."""

import json
import logging
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
from scipy.ndimage import median_filter

from . import config
from .sampler import get_video_duration, sample_evenly_spaced_frames
from .utils import clip_artifacts_dir, format_timestamp

log = logging.getLogger("pov_qa.hands_qa")


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize and optionally apply CLAHE for lighting robustness."""
    h, w = frame.shape[:2]
    target_w = config.HANDS_PREPROCESS_WIDTH
    if w != target_w:
        scale = target_w / w
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if not config.HANDS_ENABLE_CLAHE:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _skin_tone_fallback(frame: np.ndarray) -> bool:
    """Detect skin-tone regions as fallback when MediaPipe is uncertain."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(config.SKIN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(config.SKIN_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.count_nonzero(mask) / mask.size
    return skin_ratio >= config.SKIN_MIN_AREA_RATIO


def _ensure_model_asset() -> None:
    model_path = config.HAND_LANDMARKER_MODEL
    if model_path.exists():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading MediaPipe hand model to %s", model_path)
    with urlopen(config.HAND_LANDMARKER_MODEL_URL) as response:
        model_path.write_bytes(response.read())


def _region_bucket(x: float, y: float) -> dict:
    if x < 0.33:
        horizontal = "left"
    elif x < 0.67:
        horizontal = "center"
    else:
        horizontal = "right"

    if y < 0.33:
        vertical = "upper"
    elif y < 0.67:
        vertical = "middle"
    else:
        vertical = "lower"

    return {"horizontal": horizontal, "vertical": vertical}


def _detect_hands_stream(frames_iter) -> list[dict]:
    """Run MediaPipe HandLandmarker on sampled frame stream.

    Returns per-frame results with raw detection info.
    """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    _ensure_model_asset()
    model_path = str(config.HAND_LANDMARKER_MODEL)
    running_mode = (VisionRunningMode.VIDEO
                    if config.HANDS_USE_VIDEO_MODE
                    else VisionRunningMode.IMAGE)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_hands=config.MEDIAPIPE_MAX_NUM_HANDS,
        min_hand_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MEDIAPIPE_MIN_PRESENCE_CONFIDENCE,
    )

    results = []
    detector = HandLandmarker.create_from_options(options)

    try:
        last_ts_ms = -1
        for ts, frame in frames_iter:
            preprocessed = _preprocess_frame(frame)
            rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if running_mode == VisionRunningMode.VIDEO:
                ts_ms = max(int(round(ts * 1000.0)), last_ts_ms + 1)
                detection = detector.detect_for_video(mp_image, ts_ms)
                last_ts_ms = ts_ms
            else:
                detection = detector.detect(mp_image)

            hand_count = (
                len(detection.hand_landmarks)
                if detection.hand_landmarks is not None
                else 0
            )
            centroids = []
            if detection.hand_landmarks:
                for hand_landmarks in detection.hand_landmarks:
                    xs = [lm.x for lm in hand_landmarks]
                    ys = [lm.y for lm in hand_landmarks]
                    centroids.append({
                        "x": round(float(np.mean(xs)), 4),
                        "y": round(float(np.mean(ys)), 4),
                    })
            mp_detected = hand_count > 0

            max_confidence = 0.0
            if detection.handedness:
                for hand_class in detection.handedness:
                    for c in hand_class:
                        max_confidence = max(max_confidence, c.score)

            skin_present = False
            # Skin fallback is only meaningful for ">=1 hand" policy.
            if config.HANDS_REQUIRED_COUNT <= 1 and not mp_detected:
                skin_present = _skin_tone_fallback(preprocessed)

            frame_centroid = None
            region_bucket = None
            if centroids:
                frame_centroid = {
                    "x": round(float(np.mean([c["x"] for c in centroids])), 4),
                    "y": round(float(np.mean([c["y"] for c in centroids])), 4),
                }
                region_bucket = _region_bucket(frame_centroid["x"], frame_centroid["y"])

            results.append({
                "timestamp": round(ts, 3),
                "hand_count": int(hand_count),
                "mp_detected": mp_detected,
                "mp_confidence": round(max_confidence, 3),
                "hand_centroids": centroids,
                "frame_centroid": frame_centroid,
                "region_bucket": region_bucket,
                "skin_fallback": skin_present,
                "hands_present_raw": hand_count >= int(config.HANDS_REQUIRED_COUNT),
            })
    finally:
        detector.close()

    return results


def _apply_smoothing(raw_detections: list[bool]) -> list[bool]:
    """Temporal median filter to fill single-frame false negatives."""
    if len(raw_detections) < config.HANDS_SMOOTHING_WINDOW:
        return raw_detections
    arr = np.array(raw_detections, dtype=np.float32)
    smoothed = median_filter(arr, size=config.HANDS_SMOOTHING_WINDOW)
    return [bool(v >= 0.5) for v in smoothed]


def _find_missing_segments(timestamps: list[float],
                           hands_present: list[bool]) -> list[dict]:
    """Identify contiguous segments where hands are not detected."""
    segments = []
    in_gap = False
    gap_start = 0.0

    for ts, present in zip(timestamps, hands_present):
        if not present and not in_gap:
            in_gap = True
            gap_start = ts
        elif present and in_gap:
            in_gap = False
            segments.append({
                "start_time": round(gap_start, 2),
                "end_time": round(ts, 2),
                "start_fmt": format_timestamp(gap_start),
                "end_fmt": format_timestamp(ts),
            })
    if in_gap:
        segments.append({
            "start_time": round(gap_start, 2),
            "end_time": round(timestamps[-1], 2),
            "start_fmt": format_timestamp(gap_start),
            "end_fmt": format_timestamp(timestamps[-1]),
        })
    return segments


def run_hands_qa(video_path: Path,
                 artifacts_dir: Path | None = None,
                 sample_fps: float | None = None) -> dict:
    """Run hands-visible QA on a single clip.

    Returns report dict (also saved to artifacts).
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    out_dir = clip_artifacts_dir(video_path.name, artifacts_dir)
    duration = get_video_duration(video_path)
    if sample_fps is not None:
        fps = float(sample_fps)
    else:
        fps = float(config.HANDS_SAMPLE_FPS)
        if config.HANDS_DYNAMIC_SAMPLE_FPS and duration > 0:
            dynamic_fps = config.HANDS_TARGET_FRAMES / duration
            fps = min(
                float(config.HANDS_SAMPLE_FPS),
                max(float(config.HANDS_MIN_SAMPLE_FPS), float(dynamic_fps)),
            )

    report_path = _per_fps_report_path(out_dir, fps)
    if config.USE_CACHED_REPORTS and report_path.exists():
        try:
            with open(report_path) as f:
                cached = json.load(f)
            if float(cached.get("sample_fps", -1)) == float(fps):
                log.info("Hands QA cache hit: %s at %.1f fps", video_path.name, fps)
                return cached
        except Exception:
            pass

    log.info("Hands QA: %s at %.1f fps", video_path.name, fps)

    # CEO-guided fast path: evaluate evenly sampled image frames, not all frames.
    sampled_count = max(1, int(round(duration * fps))) if duration > 0 else int(max(1, fps))
    sampled_count = min(sampled_count, int(config.HANDS_TARGET_FRAMES))
    sampled_frames = sample_evenly_spaced_frames(
        video_path,
        count=sampled_count,
        max_dim=config.HANDS_PREPROCESS_WIDTH,
    )
    per_frame = _detect_hands_stream(iter(sampled_frames))

    if not per_frame:
        report = _empty_report(video_path.name, fps)
        _save_report(report, out_dir, fps)
        return report

    raw_detections = [r["hands_present_raw"] for r in per_frame]
    smoothed = _apply_smoothing(raw_detections)

    # Apply skin-tone fallback only for >=1-hand policy.
    final_detections = []
    for i, (sm, r) in enumerate(zip(smoothed, per_frame)):
        if sm:
            final_detections.append(True)
        elif (config.HANDS_REQUIRED_COUNT <= 1
              and r["skin_fallback"]
              and r["mp_confidence"] < 0.3):
            final_detections.append(True)
            per_frame[i]["recovered_by_skin"] = True
        else:
            final_detections.append(False)

    frames_with_hands = sum(final_detections)
    frames_sampled = len(final_detections)
    hands_ratio = frames_with_hands / frames_sampled if frames_sampled else 0.0

    timestamps = [r["timestamp"] for r in per_frame]
    missing_segments = _find_missing_segments(timestamps, final_detections)
    confidence_values = [float(r.get("mp_confidence", 0.0)) for r in per_frame]
    hand_presence_score = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )

    motion_steps = []
    last_centroid = None
    horizontal_counts = {"left": 0, "center": 0, "right": 0}
    vertical_counts = {"upper": 0, "middle": 0, "lower": 0}
    lower_center_hits = 0
    centroid_frames = 0
    count_buckets = {"0": 0, "1": 0, "2_plus": 0}

    for record in per_frame:
        hand_count = int(record.get("hand_count", 0))
        if hand_count <= 0:
            count_buckets["0"] += 1
        elif hand_count == 1:
            count_buckets["1"] += 1
        else:
            count_buckets["2_plus"] += 1

        centroid = record.get("frame_centroid")
        if centroid is None:
            continue

        centroid_frames += 1
        bucket = record.get("region_bucket") or _region_bucket(centroid["x"], centroid["y"])
        horizontal_counts[bucket["horizontal"]] += 1
        vertical_counts[bucket["vertical"]] += 1
        if bucket["horizontal"] == "center" and bucket["vertical"] == "lower":
            lower_center_hits += 1

        if last_centroid is not None:
            motion_steps.append(
                float(np.hypot(
                    centroid["x"] - last_centroid["x"],
                    centroid["y"] - last_centroid["y"],
                ))
            )
        last_centroid = centroid

    hand_motion_score = 0.0
    if motion_steps:
        hand_motion_score = min(
            1.0,
            float(np.mean(motion_steps)) / float(config.HAND_MOTION_NORMALIZER),
        )

    hand_count_distribution = {
        key: round(value / frames_sampled, 4) if frames_sampled else 0.0
        for key, value in count_buckets.items()
    }
    hand_region_distribution = {
        "horizontal": {
            key: round(value / centroid_frames, 4) if centroid_frames else 0.0
            for key, value in horizontal_counts.items()
        },
        "vertical": {
            key: round(value / centroid_frames, 4) if centroid_frames else 0.0
            for key, value in vertical_counts.items()
        },
        "lower_center_ratio": round(
            lower_center_hits / centroid_frames, 4
        ) if centroid_frames else 0.0,
    }
    egocentric_proxy_score = min(
        1.0,
        (0.45 * hands_ratio)
        + (0.30 * hand_region_distribution["vertical"]["lower"])
        + (0.25 * hand_region_distribution["horizontal"]["center"]),
    )

    report = {
        "filename": video_path.name,
        "required_hands": int(config.HANDS_REQUIRED_COUNT),
        "sample_fps": fps,
        "frames_sampled": frames_sampled,
        "frames_with_hands": frames_with_hands,
        "hands_ratio": round(hands_ratio, 4),
        "hand_visible_ratio": round(hands_ratio, 4),
        "hand_presence_score": round(hand_presence_score, 4),
        "hand_motion_score": round(hand_motion_score, 4),
        "hand_count_distribution": hand_count_distribution,
        "hand_region_distribution": hand_region_distribution,
        "egocentric_proxy_score": round(egocentric_proxy_score, 4),
        "missing_segments": missing_segments,
        "sampling_strategy": "evenly_spaced_frames",
        "sampled_frame_target": sampled_count,
        "smoothing_window": config.HANDS_SMOOTHING_WINDOW,
        "per_frame_detail": per_frame,
    }

    # Resample-on-borderline logic
    if (config.HANDS_BORDERLINE_LOW <= hands_ratio < config.HANDS_BORDERLINE_HIGH
            and fps < config.HANDS_RESAMPLE_FPS):
        if duration >= config.HANDS_LONG_CLIP_NO_RESAMPLE_SEC:
            report["resample_skipped"] = True
            report["resample_skipped_reason"] = (
                f"long_clip_{duration:.1f}s_manual_borderline_review"
            )
            _save_report(report, out_dir, fps)
            return report

        # Cache the current fps result to avoid recomputing on repeated runs.
        _save_report(report, out_dir, fps, update_latest=False)
        log.info("Borderline hands_ratio=%.3f — resampling at %d fps",
                 hands_ratio, config.HANDS_RESAMPLE_FPS)
        resample_report = run_hands_qa(
            video_path, artifacts_dir=artifacts_dir,
            sample_fps=config.HANDS_RESAMPLE_FPS,
        )
        resample_report["resampled_from_fps"] = fps
        return resample_report

    _save_report(report, out_dir, fps)
    return report


def _empty_report(filename: str, fps: float) -> dict:
    return {
        "filename": filename,
        "required_hands": int(config.HANDS_REQUIRED_COUNT),
        "sample_fps": fps,
        "frames_sampled": 0,
        "frames_with_hands": 0,
        "hands_ratio": 0.0,
        "hand_visible_ratio": 0.0,
        "hand_presence_score": 0.0,
        "hand_motion_score": 0.0,
        "hand_count_distribution": {"0": 0.0, "1": 0.0, "2_plus": 0.0},
        "hand_region_distribution": {
            "horizontal": {"left": 0.0, "center": 0.0, "right": 0.0},
            "vertical": {"upper": 0.0, "middle": 0.0, "lower": 0.0},
            "lower_center_ratio": 0.0,
        },
        "egocentric_proxy_score": 0.0,
        "missing_segments": [],
        "error": "no frames extracted",
    }


def _per_fps_report_path(out_dir: Path, fps: float) -> Path:
    fps_str = f"{fps:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    return out_dir / f"hands_report_{fps_str}fps.json"


def _save_report(report: dict, out_dir: Path, fps: float,
                 update_latest: bool = True) -> None:
    # Strip per-frame detail from saved file to keep it manageable
    save_data = {k: v for k, v in report.items() if k != "per_frame_detail"}
    with open(_per_fps_report_path(out_dir, fps), "w") as f:
        json.dump(save_data, f, indent=2)
    if update_latest:
        with open(out_dir / "hands_report.json", "w") as f:
            json.dump(save_data, f, indent=2)
    else:
        # Ensure the canonical report exists for downstream steps on first run.
        latest = out_dir / "hands_report.json"
        if not latest.exists():
            with open(latest, "w") as f:
                json.dump(save_data, f, indent=2)


def check_c3a(report: dict) -> tuple[bool, str, str]:
    """Evaluate hands-visible pass/fail/borderline from hand report.

    Returns (passed, detail_string, gate_status).
    gate_status is 'pass', 'fail', or 'borderline'.
    """
    ratio = report["hands_ratio"]
    missing = report.get("missing_segments", [])

    missing_desc = ""
    if missing:
        segs = [f"{s['start_fmt']}-{s['end_fmt']}" for s in missing[:4]]
        missing_desc = ", missing around " + ", ".join(segs)

    label = "hands_visible"
    if int(config.HANDS_REQUIRED_COUNT) >= 2:
        label = "both_hands_visible"

    if ratio >= config.HANDS_MIN_RATIO:
        return True, f"{label} {ratio:.2f}", "pass"

    detail = f"{label} {ratio:.2f} (<{config.HANDS_MIN_RATIO}){missing_desc}"

    if ratio < config.HANDS_BORDERLINE_LOW:
        return False, detail, "fail"

    return False, detail, "borderline"
