"""Public dataset-curation helpers layered on top of the existing QA pipeline."""

from __future__ import annotations

import json
import math
from typing import Iterable

from . import config


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, safe_float(value, lo)))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalize_inverse(value, normalizer: float) -> float:
    raw = safe_float(value, default=0.0)
    if normalizer <= 0:
        return 0.0
    return clamp(1.0 - min(raw / normalizer, 1.0))


def split_semicolon_values(value) -> list[str]:
    if value is None:
        return []
    try:
        if value != value:
            return []
    except Exception:
        pass
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def encode_list(values: Iterable[str]) -> str:
    return "; ".join(dict.fromkeys(v for v in values if v))


def encode_json(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def summarize_quality(precheck_report: dict | None) -> dict:
    precheck_report = precheck_report or {}
    if not precheck_report:
        return {
            "brightness_score": None,
            "blur_score": None,
            "motion_score": None,
            "static_duplicate_score": None,
        }

    dark_ratio = clamp(precheck_report.get("dark_ratio", 0.0))
    blurry_ratio = clamp(precheck_report.get("blurry_ratio", 0.0))
    motion_score = clamp(precheck_report.get("motion_score", 0.0))
    static_duplicate_score = clamp(precheck_report.get("static_duplicate_score", 0.0))
    return {
        "brightness_score": clamp(1.0 - dark_ratio),
        "blur_score": clamp(1.0 - blurry_ratio),
        "motion_score": motion_score,
        "static_duplicate_score": static_duplicate_score,
    }


def compute_quality_flags(row: dict, precheck_report: dict | None = None) -> list[str]:
    flags: list[str] = []
    precheck_report = precheck_report or {}

    if not bool(row.get("c1_pass", False)):
        flags.append("low_resolution")
    if not bool(row.get("c2_pass", False)):
        flags.append("low_frame_rate")
    if row.get("c3a_gate") == "fail":
        flags.append("low_hand_visibility")
    elif row.get("c3a_gate") == "borderline":
        flags.append("hand_visibility_borderline")
    if row.get("c3b_gate") == "fail":
        flags.append("third_person_likely")
    elif row.get("c3b_gate") == "borderline":
        flags.append("camera_perspective_ambiguous")
    if row.get("c9_gate") == "fail":
        flags.append("unstable_camera")
    elif row.get("c9_gate") == "borderline":
        flags.append("camera_stability_borderline")

    if safe_float(precheck_report.get("dark_ratio")) >= config.MANUAL_DARK_RATIO_FLAG:
        flags.append("low_brightness")
    if safe_float(precheck_report.get("blurry_ratio")) >= config.MANUAL_BLURRY_RATIO_FLAG:
        flags.append("blur")
    if safe_float(precheck_report.get("static_duplicate_score")) >= config.STATIC_DUPLICATE_FLAG_SCORE:
        flags.append("static_or_duplicate")
    return list(dict.fromkeys(flags))


def compute_curation_reasons(row: dict, quality_flags: Iterable[str]) -> list[str]:
    reasons = split_semicolon_values(row.get("reject_reasons", ""))
    if row.get("gate_result") == config.GATE_BORDERLINE:
        reasons.append("manual_review_required")
    elif row.get("gate_result") == config.GATE_FAIL:
        reasons.append("automated_quality_filter")

    if "third_person_likely" in quality_flags:
        reasons.append("non_egocentric_view")
    if "static_or_duplicate" in quality_flags:
        reasons.append("limited_training_novelty")
    return list(dict.fromkeys(reasons))


def compute_training_value_components(row: dict, precheck_report: dict | None = None) -> dict:
    quality = summarize_quality(precheck_report)
    hand_visible_ratio = row.get("hand_visible_ratio")
    if hand_visible_ratio is None:
        hand_visible_ratio = row.get("c3a_hands_ratio")
    components = {
        "hand_visible_ratio": None if hand_visible_ratio is None else clamp(safe_float(hand_visible_ratio, 0.0)),
        "hand_presence_score": None if row.get("hand_presence_score") is None else clamp(row.get("hand_presence_score", 0.0)),
        "hand_motion_score": None if row.get("hand_motion_score") is None else clamp(row.get("hand_motion_score", 0.0)),
        "egocentric_proxy_score": None if row.get("egocentric_proxy_score") is None else clamp(row.get("egocentric_proxy_score", 0.0)),
        "brightness_score": quality["brightness_score"],
        "blur_score": quality["blur_score"],
        "motion_score": quality["motion_score"],
        "camera_stability_score": (
            None
            if row.get("c9_max_motion") is None
            else normalize_inverse(row.get("c9_max_motion", 0.0), config.CAMERA_STABILITY_NORMALIZER)
        ),
        "static_duplicate_score": (
            None
            if quality["static_duplicate_score"] is None
            else clamp(1.0 - quality["static_duplicate_score"])
        ),
    }
    return components


def compute_training_value_score(row: dict, precheck_report: dict | None = None) -> tuple[float, dict]:
    components = compute_training_value_components(row, precheck_report=precheck_report)
    score = 0.0
    total_weight = 0.0
    for key, weight in config.TRAINING_VALUE_WEIGHTS.items():
        value = components.get(key)
        if value is None:
            continue
        score += clamp(value) * float(weight)
        total_weight += float(weight)
    if total_weight > 0:
        score /= total_weight
    penalties = {
        "low_resolution": 0.28,
        "low_frame_rate": 0.22,
        "third_person_likely": 0.18,
        "unstable_camera": 0.18,
        "low_brightness": 0.12,
        "blur": 0.12,
        "static_or_duplicate": 0.12,
    }
    for flag in compute_quality_flags(row, precheck_report=precheck_report):
        score -= penalties.get(flag, 0.0)
    return round(clamp(score), 4), components


def public_status_from_row(row: dict,
                           training_value_score: float,
                           quality_flags: Iterable[str]) -> str:
    flags = set(quality_flags)
    gate_result = row.get("gate_result")
    if gate_result == config.GATE_FAIL:
        return config.PUBLIC_STATUS_LOW_VALUE
    if gate_result == config.GATE_BORDERLINE:
        return config.PUBLIC_STATUS_NEEDS_REVIEW
    if training_value_score <= safe_float(config.STATUS_THRESHOLDS.get("low_value"), 0.38):
        return config.PUBLIC_STATUS_LOW_VALUE
    if "third_person_likely" in flags or "static_or_duplicate" in flags:
        return config.PUBLIC_STATUS_LOW_VALUE
    if training_value_score >= safe_float(config.STATUS_THRESHOLDS.get("recommended"), 0.68):
        return config.PUBLIC_STATUS_RECOMMENDED
    return config.PUBLIC_STATUS_NEEDS_REVIEW
