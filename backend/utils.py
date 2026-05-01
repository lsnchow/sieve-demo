"""Shared helpers: logging, frame I/O, path management, and input discovery."""

import hashlib
import logging
import os
from pathlib import Path

from . import config


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("pov_qa")


def configure_runtime_threads() -> None:
    """Optionally cap CV/BLAS thread usage via POV_QA_THREADS env var."""
    val = os.getenv("POV_QA_THREADS", "").strip()
    if not val:
        return
    try:
        n = int(val)
    except ValueError:
        return
    if n <= 0:
        return

    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

    try:
        import cv2
        cv2.setNumThreads(n)
    except Exception:
        pass


def clip_artifacts_dir(filename: str, base: Path | None = None) -> Path:
    base = base or config.ARTIFACTS_DIR
    d = base / Path(filename).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def md5_hash(filepath: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def collect_video_files(input_dir: Path) -> list[Path]:
    """Collect supported video files recursively, excluding hidden files."""
    files = [
        p for p in input_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in config.VIDEO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: (str(p.parent), p.name))


def infer_demo_category(video_path: Path, input_dir: Path) -> str:
    """Infer demo_category from the first folder below input_dir when present."""
    try:
        relative = video_path.relative_to(input_dir)
    except ValueError:
        return ""

    if len(relative.parts) <= 1:
        return ""
    return relative.parts[0]
