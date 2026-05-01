"""Phase A: ingest source clips, ffprobe metadata, and provenance sidecar fields."""

import csv
import json
import logging
import subprocess
from pathlib import Path

from tqdm import tqdm

from . import config
from .utils import clip_artifacts_dir, collect_video_files, infer_demo_category, md5_hash

log = logging.getLogger("pov_qa.ingest")


def _load_source_map(input_dir: Path) -> dict[str, dict]:
    source_map_path = input_dir / config.DEMO_SOURCE_MAP_NAME
    if not source_map_path.exists():
        return {}

    with open(source_map_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            filename = (row.get("filename") or "").strip()
            relative_path = (row.get("relative_path") or "").strip()
            key = relative_path or filename
            if not key:
                continue
            if filename:
                cleaned = {}
                for k, v in row.items():
                    if k is None:
                        continue
                    if isinstance(v, list):
                        cleaned[k] = " ".join(str(part).strip() for part in v if str(part).strip())
                    else:
                        cleaned[k] = str(v or "").strip()
                rows[key] = cleaned
        return rows


def _ffprobe(filepath: Path) -> dict:
    """Run ffprobe and return parsed JSON."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(filepath),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _extract_metadata(filepath: Path) -> dict:
    """Extract resolution, fps, duration, rotation from ffprobe output."""
    probe = _ffprobe(filepath)

    video_stream = None
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {filepath.name}")

    raw_w = int(video_stream.get("width", 0))
    raw_h = int(video_stream.get("height", 0))

    rotation = 0
    side_data = video_stream.get("side_data_list", [])
    for sd in side_data:
        if "rotation" in sd:
            rotation = abs(int(sd["rotation"]))
            break
    if rotation == 0:
        rot_tag = video_stream.get("tags", {}).get("rotate", "0")
        rotation = abs(int(rot_tag))

    if rotation in (90, 270):
        display_w, display_h = raw_h, raw_w
    else:
        display_w, display_h = raw_w, raw_h

    fmt = probe.get("format", {})
    duration = float(fmt.get("duration", 0))

    nb_frames_str = video_stream.get("nb_frames", "0")
    nb_frames = int(nb_frames_str) if nb_frames_str != "N/A" else 0

    if nb_frames > 0 and duration > 0:
        effective_fps = nb_frames / duration
    else:
        r_rate = video_stream.get("r_frame_rate", "30/1")
        num, den = r_rate.split("/")
        effective_fps = float(num) / float(den) if float(den) != 0 else 0.0

    return {
        "filename": filepath.name,
        "source_path": str(filepath.resolve()),
        "raw_w": raw_w,
        "raw_h": raw_h,
        "rotation": rotation,
        "display_w": display_w,
        "display_h": display_h,
        "duration": round(duration, 3),
        "nb_frames": nb_frames,
        "effective_fps": round(effective_fps, 2),
        "r_frame_rate": video_stream.get("r_frame_rate", ""),
        "codec": video_stream.get("codec_name", ""),
        "file_size_bytes": int(fmt.get("size", 0)),
    }


def check_c1(meta: dict) -> tuple[bool, str]:
    """Resolution threshold after rotation."""
    w, h = meta["display_w"], meta["display_h"]
    if w >= config.MIN_WIDTH and h >= config.MIN_HEIGHT:
        return True, f"resolution {w}x{h}"
    return False, f"resolution {w}x{h} (<{config.MIN_WIDTH}x{config.MIN_HEIGHT})"


def check_c2(meta: dict) -> tuple[bool, str]:
    """Frame-rate threshold."""
    fps = meta["effective_fps"]
    if fps >= config.MIN_FPS:
        return True, f"fps {fps:.1f}"
    return False, f"fps {fps:.1f} (<{config.MIN_FPS:.0f})"


def run_ingest(input_dir: Path, expected_count: int | None = None,
               artifacts_dir: Path | None = None) -> list[dict]:
    """Enumerate clips, extract metadata, compute hashes, and run format checks.

    Returns list of per-clip metadata dicts (also saved to artifacts).
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    files = collect_video_files(input_dir)

    if expected_count is not None and len(files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} clips, found {len(files)} in {input_dir}"
        )

    log.info("Found %d clips in %s", len(files), input_dir)
    source_map = _load_source_map(input_dir)

    results = []
    for fp in tqdm(files, desc="Ingesting clips"):
        log.debug("Processing %s", fp.name)

        meta = _extract_metadata(fp)
        meta["md5"] = md5_hash(fp)
        relative_key = str(fp.relative_to(input_dir))
        source_meta = source_map.get(relative_key, source_map.get(fp.name, {}))
        meta["source_url"] = source_meta.get("source_url", "")
        meta["license"] = source_meta.get("license", "")
        meta["attribution"] = source_meta.get("attribution", "")
        meta["demo_category"] = source_meta.get("demo_category", infer_demo_category(fp, input_dir))
        meta["source_title"] = source_meta.get("source_title", fp.stem)
        meta["clip_start_sec"] = source_meta.get("clip_start_sec", "")
        meta["clip_duration_sec"] = source_meta.get("clip_duration_sec", "")

        c1_pass, c1_detail = check_c1(meta)
        c2_pass, c2_detail = check_c2(meta)
        meta["c1_pass"] = c1_pass
        meta["c1_detail"] = c1_detail
        meta["c2_pass"] = c2_pass
        meta["c2_detail"] = c2_detail

        out_dir = clip_artifacts_dir(fp.name, artifacts_dir)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)
        log.info(
            "%s | %dx%d | %.1f fps | %.1fs | resolution=%s fps=%s",
            fp.name, meta["display_w"], meta["display_h"],
            meta["effective_fps"], meta["duration"],
            "PASS" if c1_pass else "FAIL",
            "PASS" if c2_pass else "FAIL",
        )

    return results
