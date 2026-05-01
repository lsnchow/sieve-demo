"""Wide-lens QA via barrel-distortion and line-curvature analysis."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from . import config
from .sampler import sample_evenly_spaced_frames
from .utils import clip_artifacts_dir

log = logging.getLogger("pov_qa.wide_lens_qa")


def _nearest_edge_offset_on_normal(edges: np.ndarray,
                                   base_x: float,
                                   base_y: float,
                                   nx: float,
                                   ny: float,
                                   max_offset: int) -> int | None:
    """Find nearest edge pixel offset along the local normal direction."""
    h, w = edges.shape
    best = None
    for off in range(-max_offset, max_offset + 1):
        px = int(round(base_x + nx * off))
        py = int(round(base_y + ny * off))
        if px < 0 or px >= w or py < 0 or py >= h:
            continue
        if edges[py, px] > 0:
            if best is None or abs(off) < abs(best):
                best = off
                if off == 0:
                    break
    return best


def _compute_line_curvature(frame: np.ndarray) -> dict:
    """Measure straight-line bending using sampled normal offsets on Hough lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 0),
        int(config.WIDE_LENS_CANNY_LOW),
        int(config.WIDE_LENS_CANNY_HIGH),
    )
    min_len = max(int(w * float(config.WIDE_LENS_LINE_MIN_LENGTH_RATIO)), 40)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=int(config.WIDE_LENS_HOUGH_THRESHOLD),
        minLineLength=min_len,
        maxLineGap=int(config.WIDE_LENS_LINE_MAX_GAP),
    )

    if lines is None or len(lines) < 2:
        return {"curvature_score": 0.0, "num_lines": 0, "confidence": 0.0}

    max_dist = float(np.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2))
    per_line_scores: list[float] = []
    analyzed_lines = 0

    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue

        ux = (x2 - x1) / length
        uy = (y2 - y1) / length
        nx = -uy
        ny = ux

        sample_step = max(int(config.WIDE_LENS_LINE_SAMPLE_STEP_PX), 2)
        sample_count = max(int(length / sample_step), 8)
        offsets = []
        for t in np.linspace(0.0, length, sample_count):
            bx = x1 + ux * t
            by = y1 + uy * t
            off = _nearest_edge_offset_on_normal(
                edges,
                base_x=bx,
                base_y=by,
                nx=nx,
                ny=ny,
                max_offset=int(config.WIDE_LENS_LINE_SCAN_NORMAL_PX),
            )
            if off is not None:
                offsets.append(float(off))

        if len(offsets) < max(6, sample_count // 3):
            continue

        arr = np.asarray(offsets, dtype=np.float32)
        idx = np.linspace(-1.0, 1.0, len(arr), dtype=np.float32)
        if len(arr) >= 3:
            # Remove tilt-induced drift; keep non-linearity only.
            coeff = np.polyfit(idx, arr, 1)
            trend = coeff[0] * idx + coeff[1]
            residual = arr - trend
        else:
            residual = arr - float(np.mean(arr))

        bend_px = float(np.sqrt(np.mean(np.square(residual))))
        bend_norm = float(np.clip(
            bend_px / max(float(config.WIDE_LENS_LINE_SCAN_NORMAL_PX), 1.0), 0.0, 1.0
        ))

        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        dist_norm = float(np.hypot(mid_x - w / 2.0, mid_y - h / 2.0) / max_dist)
        peripheral_weight = float(np.clip((dist_norm - 0.25) / 0.75, 0.0, 1.0))
        length_weight = min(length / (w * 0.45), 1.0)

        score = bend_norm * (0.6 + 0.4 * peripheral_weight) * length_weight
        per_line_scores.append(float(np.clip(score, 0.0, 1.0)))
        analyzed_lines += 1

    if not per_line_scores:
        return {"curvature_score": 0.0, "num_lines": 0, "confidence": 0.0}

    curvature_score = float(np.clip(np.mean(per_line_scores), 0.0, 1.0))
    confidence = min(
        analyzed_lines / max(float(config.WIDE_LENS_MIN_LINES_FOR_CONF), 1.0),
        1.0,
    )

    return {
        "curvature_score": round(curvature_score, 4),
        "num_lines": int(analyzed_lines),
        "confidence": round(float(confidence), 4),
    }


def _compute_edge_density_falloff(frame: np.ndarray) -> float:
    """Wide-lens cameras show vignetting: edge density drops toward periphery.
    Standard lenses maintain more uniform edge density.

    Returns a falloff ratio (higher = more wide-lens-like).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    # Center region vs periphery ring
    margin_x, margin_y = w // 4, h // 4
    center = edges[margin_y:h - margin_y, margin_x:w - margin_x]
    center_density = np.mean(center > 0) if center.size else 0

    # Periphery: the full frame minus center
    mask = np.ones_like(edges, dtype=bool)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = False
    periphery = edges[mask]
    periphery_density = np.mean(periphery > 0) if periphery.size else 0

    if periphery_density == 0:
        return 0.0
    return center_density / max(periphery_density, 1e-6)


def _compute_radial_distortion_score(frame: np.ndarray) -> float:
    """Detect barrel distortion by checking if edge points curve away from
    radial lines emanating from the frame center.

    Wide-angle lenses produce barrel distortion where straight lines
    bow outward from center, especially near edges.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w / 2, h / 2

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    # Find edge point coordinates in the outer 30% ring of the frame
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    inner_r = max_r * 0.5

    ey, ex = np.nonzero(edges)
    if len(ey) < 50:
        return 0.0

    dx = ex.astype(float) - cx
    dy = ey.astype(float) - cy
    r = np.sqrt(dx ** 2 + dy ** 2)

    outer_mask = r > inner_r
    if np.sum(outer_mask) < 20:
        return 0.0

    # In barrel distortion, lines along the periphery systematically curve.
    # Measure by looking at local edge orientation vs radial direction.
    # Compute gradient orientation
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_angle = np.arctan2(sobel_y, sobel_x)

    outer_ey, outer_ex = ey[outer_mask], ex[outer_mask]
    outer_dx = outer_ex.astype(float) - cx
    outer_dy = outer_ey.astype(float) - cy
    radial_angle = np.arctan2(outer_dy, outer_dx)

    grad_at_points = grad_angle[outer_ey, outer_ex]
    angle_diff = np.abs(np.cos(grad_at_points - radial_angle))

    # With barrel distortion, gradients at edges tend to be more tangential
    # (perpendicular to radial direction), so cos(diff) is low.
    tangential_score = 1.0 - np.mean(angle_diff)
    return float(np.clip(tangential_score, 0, 1))


def _score_single_frame(frame: np.ndarray) -> dict:
    """Compute all wide-lens indicators for one frame."""
    edge_falloff = _compute_edge_density_falloff(frame)
    radial_score = _compute_radial_distortion_score(frame)
    line_info = _compute_line_curvature(frame)

    edge_norm = float(np.clip((edge_falloff - 1.0) / 2.0, 0.0, 1.0))
    line_strength = float(np.clip(
        0.7 * line_info["curvature_score"] + 0.3 * line_info["confidence"], 0.0, 1.0
    ))

    # Composite score: generic geometric evidence (no clip-specific rules).
    composite = (
        float(config.WIDE_LENS_WEIGHT_RADIAL) * radial_score +
        float(config.WIDE_LENS_WEIGHT_EDGE) * edge_norm +
        float(config.WIDE_LENS_WEIGHT_LINE) * line_strength
    )

    return {
        "radial_distortion": round(radial_score, 4),
        "edge_density_falloff": round(edge_falloff, 4),
        "line_curvature": line_info,
        "composite": round(composite, 4),
    }


def run_wide_lens_qa(video_path: Path,
                     artifacts_dir: Path | None = None) -> dict:
    """Analyze a clip for wide-lens characteristics.

    Returns report dict (also saved to artifacts).
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    out_dir = clip_artifacts_dir(video_path.name, artifacts_dir)
    report_path = out_dir / "wide_lens_report.json"

    if config.USE_CACHED_REPORTS and report_path.exists():
        try:
            with open(report_path) as f:
                cached = json.load(f)
            if cached.get("num_frames_analyzed", 0) > 0:
                log.info("Wide-lens QA cache hit: %s", video_path.name)
                return cached
        except Exception:
            pass

    log.info("Wide-lens QA: %s", video_path.name)

    # Sample fewer frames — wide-lens is a property of the lens, not temporal
    n = config.WIDE_LENS_SAMPLE_FRAMES
    frames = sample_evenly_spaced_frames(
        video_path, count=n, max_dim=int(config.WIDE_LENS_MAX_DIM)
    )

    if not frames:
        report = {
            "filename": video_path.name,
            "wide_lens_score": 0.0,
            "confidence": 0.0,
            "num_frames_analyzed": 0,
            "error": "no frames extracted",
        }
        _save_report(report, out_dir)
        return report

    per_frame_scores = []
    for ts, frame in frames:
        score = _score_single_frame(frame)
        score["timestamp"] = round(ts, 2)
        per_frame_scores.append(score)

    composites = [s["composite"] for s in per_frame_scores]
    wide_lens_score = float(np.median(composites))
    mean_radial = float(np.mean([s["radial_distortion"] for s in per_frame_scores]))
    mean_edge_falloff = float(np.mean([s["edge_density_falloff"] for s in per_frame_scores]))
    mean_line_conf = float(np.mean([s["line_curvature"]["confidence"] for s in per_frame_scores]))

    # Confidence based on consistency across frames
    std = float(np.std(composites))
    consistency = max(0, 1.0 - std * 3)
    confidence = consistency * min(len(frames) / n, 1.0)

    # Save example frames for human review
    example_dir = out_dir / "wide_lens_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    example_paths = []
    for i, (ts, frame) in enumerate(frames[:4]):
        path = example_dir / f"wl_sample_{i:02d}_{ts:.1f}s.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        example_paths.append(str(path))

    report = {
        "filename": video_path.name,
        "wide_lens_score": round(wide_lens_score, 4),
        "mean_radial_distortion": round(mean_radial, 4),
        "mean_edge_falloff": round(mean_edge_falloff, 4),
        "mean_line_curvature": round(
            float(np.mean([s["line_curvature"]["curvature_score"] for s in per_frame_scores])),
            4,
        ),
        "mean_line_confidence": round(mean_line_conf, 4),
        "confidence": round(confidence, 4),
        "num_frames_analyzed": len(frames),
        "per_frame_scores": per_frame_scores,
        "example_frames": example_paths,
    }

    _save_report(report, out_dir)
    return report


def _save_report(report: dict, out_dir: Path) -> None:
    save_data = {k: v for k, v in report.items() if k != "per_frame_scores"}
    with open(out_dir / "wide_lens_report.json", "w") as f:
        json.dump(save_data, f, indent=2)


def check_c3b(report: dict) -> tuple[bool, str, str]:
    """Evaluate perspective fit from the wide-lens report.

    Returns (passed, detail_string, gate_status).
    """
    score = report["wide_lens_score"]
    edge_falloff = float(report.get("mean_edge_falloff", 0.0))
    line_conf = float(report.get("mean_line_confidence", 0.0))
    conf = report["confidence"]

    if score >= config.WIDE_LENS_PASS_THRESHOLD and conf >= config.WIDE_LENS_CONFIDENCE_MIN:
        return True, f"wide_lens_score={score:.2f}", "pass"

    # Low-score salvage path: only used below borderline-low threshold and only
    # when confidence is still high enough to avoid false wide-lens passes.
    if score < config.WIDE_LENS_BORDERLINE_LOW:
        if (
            score >= config.WIDE_LENS_SALVAGE_SCORE_MIN
            and conf >= config.WIDE_LENS_SALVAGE_CONFIDENCE_MIN
            and (
                edge_falloff >= config.WIDE_LENS_SALVAGE_EDGE_FALLOFF_MIN
                or line_conf >= config.WIDE_LENS_SALVAGE_LINE_CONF_MIN
            )
        ):
            return (
                True,
                (
                    f"wide_lens_score={score:.2f} "
                    f"(salvaged by edge_falloff={edge_falloff:.2f}, "
                    f"line_conf={line_conf:.2f}, confidence={conf:.2f})"
                ),
                "pass",
            )
        detail = f"non-wide-lens framing (wide_lens_score={score:.2f})"
        return False, detail, "fail"

    # Very low confidence at modest score is likely non-wide-lens framing.
    if (score <= config.WIDE_LENS_LOW_CONF_FAIL_SCORE_MAX
            and conf <= config.WIDE_LENS_LOW_CONF_FAIL_CONF_MAX):
        detail = (
            f"non-wide-lens low-confidence framing "
            f"(wide_lens_score={score:.2f}, confidence={conf:.2f})"
        )
        return False, detail, "fail"

    # Close-up / narrow-FOV heuristic: if confidence is low and score is still low,
    # treat as non-wide-lens instead of borderline.
    if score < config.WIDE_LENS_CLOSEUP_SCORE_MAX and conf <= config.WIDE_LENS_CLOSEUP_CONF_MAX:
        detail = (
            f"non-wide-lens close-up framing "
            f"(wide_lens_score={score:.2f}, confidence={conf:.2f})"
        )
        return False, detail, "fail"

    detail = f"wide-lens uncertain (wide_lens_score={score:.2f}, confidence={conf:.2f})"
    return False, detail, "borderline"
