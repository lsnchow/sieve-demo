"""Frame sampling utilities for sequential sampling + keyframe generation."""

import logging
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger("pov_qa.sampler")


def _resize_if_needed(frame: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def iter_sampled_frames(video_path: Path, sample_fps: float = None,
                        max_dim: int | None = None):
    """Yield sampled frames as (timestamp_seconds, frame_bgr).

    Uses sequential decode + frame skipping, which is much faster than random seek
    on long clips.
    """
    sample_fps = sample_fps or config.HANDS_SAMPLE_FPS
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if video_fps <= 0:
            video_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0.0

        step = max(1, int(round(video_fps / sample_fps)))
        frame_idx = 0
        yielded = 0

        while True:
            # Skip intermediate frames without materializing them as arrays.
            for _ in range(step - 1):
                if not cap.grab():
                    return
                frame_idx += 1

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            ts = frame_idx / video_fps
            frame_idx += 1

            yield ts, _resize_if_needed(frame, max_dim)
            yielded += 1

        log.debug("%s: sampled %d frames at %.2f fps (duration %.1fs)",
                  video_path.name, yielded, sample_fps, duration)
    finally:
        cap.release()


def sample_frames(video_path: Path, sample_fps: float = None,
                  max_dim: int | None = None) -> list[tuple[float, np.ndarray]]:
    """Materialize sampled frames into a list.

    Prefer `iter_sampled_frames` for long clips to avoid high memory usage.
    """
    return list(iter_sampled_frames(video_path, sample_fps=sample_fps, max_dim=max_dim))


def sample_evenly_spaced_frames(video_path: Path, count: int,
                                max_dim: int | None = None) -> list[tuple[float, np.ndarray]]:
    """Sample a fixed number of evenly spaced frames via sparse seeks."""
    if count <= 0:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if video_fps <= 0:
            video_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if total_frames > 0 else 0.0

        timestamps = np.linspace(0, max(duration - 0.05, 0), count)
        sampled = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            sampled.append((float(ts), _resize_if_needed(frame, max_dim)))
        return sampled
    finally:
        cap.release()


def extract_keyframes(video_path: Path, count: int = None,
                      output_dir: Path | None = None) -> list[Path]:
    """Extract N evenly-spaced keyframes and save to output_dir.

    Returns list of saved file paths.
    """
    count = count or config.KEYFRAME_COUNT

    if output_dir is None:
        output_dir = config.ARTIFACTS_DIR / video_path.stem / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("keyframe_*.jpg"))
    if len(existing) >= count:
        return existing[:count]

    sampled = sample_evenly_spaced_frames(video_path, count=count)
    saved = []

    for i, (ts, frame) in enumerate(sampled):
        out_path = output_dir / f"keyframe_{i:03d}_{float(ts):.1f}s.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        saved.append(out_path)

    log.debug("%s: extracted %d keyframes", video_path.name, len(saved))
    return saved


def get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps > 0 else 0.0
