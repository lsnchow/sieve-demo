"""Dataset preparation utilities: validate, repair, extract, and verify clips."""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("pov_qa.dataset_prep")

CLIP_NAME_RE = re.compile(r".+_file1\.(mp4|mov)$", re.IGNORECASE)


def _run_command(cmd: list[str], input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _zip_is_valid(zip_path: Path) -> tuple[bool, str]:
    proc = _run_command(["unzip", "-tqq", str(zip_path)])
    if proc.returncode == 0:
        return True, "unzip_test_ok"
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    msg = detail[-1] if detail else f"unzip_exit_{proc.returncode}"
    return False, msg


def _bsdtar_can_list(zip_path: Path) -> tuple[bool, str]:
    proc = _run_command(["bsdtar", "-tf", str(zip_path)])
    if proc.returncode == 0:
        return True, "bsdtar_list_ok"
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    msg = detail[-1] if detail else f"bsdtar_exit_{proc.returncode}"
    return False, msg


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _disk_free_bytes(path: Path) -> int:
    resolved = path.resolve()
    usage = shutil.disk_usage(resolved)
    return int(usage.free)


def _required_free_bytes(zip_size: int, repair_mode: str) -> int:
    # Conservative guard: repair-first may need archive copy + extracted content.
    factor = 2.2 if repair_mode == "repair-first" else 1.4
    return int(zip_size * factor)


def _candidate_video_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and CLIP_NAME_RE.fullmatch(p.name)
    )


def _verify_target_inventory(target_dir: Path, expected_count: int) -> list[Path]:
    files = sorted(
        p for p in target_dir.iterdir()
        if p.is_file() and CLIP_NAME_RE.fullmatch(p.name)
    )
    if len(files) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} clips in {target_dir}, found {len(files)}"
        )
    return files


def _write_manifest(files: list[Path], target_dir: Path) -> Path:
    manifest_path = target_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "size_bytes", "sha256"])
        for fp in sorted(files, key=lambda p: p.name):
            writer.writerow([fp.name, fp.stat().st_size, _sha256(fp)])
    return manifest_path


def _attempt_repair(zip_path: Path, workspace_dir: Path) -> tuple[Path | None, list[str]]:
    attempts: list[str] = []

    repaired_f = workspace_dir / "repaired_F.zip"
    proc_f = _run_command(["zip", "-F", str(zip_path), "--out", str(repaired_f)])
    attempts.append(f"zip -F exit={proc_f.returncode}")
    if proc_f.returncode == 0 and repaired_f.exists():
        ok, detail = _zip_is_valid(repaired_f)
        attempts.append(f"validate repaired_F: {detail}")
        if ok:
            return repaired_f, attempts

    repaired_ff = workspace_dir / "repaired_FF.zip"
    proc_ff = _run_command(
        ["zip", "-FF", str(zip_path), "--out", str(repaired_ff)],
        input_text="y\n",
    )
    attempts.append(f"zip -FF exit={proc_ff.returncode}")
    if proc_ff.returncode == 0 and repaired_ff.exists():
        ok, detail = _zip_is_valid(repaired_ff)
        attempts.append(f"validate repaired_FF: {detail}")
        if ok:
            return repaired_ff, attempts

    return None, attempts


def prepare_dataset(zip_path: Path,
                    target_dir: Path,
                    repair_mode: str = "repair-first",
                    delete_source: bool = True,
                    expected_count: int = 68) -> dict:
    """Validate/repair zip, extract clips, verify inventory, and write manifest."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    if repair_mode not in {"repair-first", "validate-only"}:
        raise ValueError(f"Unsupported repair_mode: {repair_mode}")

    zip_size = int(zip_path.stat().st_size)
    target_dir = target_dir.resolve()
    target_parent = target_dir.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    required_free = _required_free_bytes(zip_size, repair_mode)
    free_bytes = _disk_free_bytes(target_parent)
    if free_bytes < required_free:
        raise RuntimeError(
            "Insufficient disk space for safe extract/repair. "
            f"free={free_bytes} required~={required_free}. "
            "Re-download to a volume with more free space."
        )

    actions = [f"zip_size_bytes={zip_size}", f"free_bytes={free_bytes}"]

    # Reuse existing extracted dataset if already complete.
    if target_dir.exists() and any(target_dir.iterdir()):
        existing = _verify_target_inventory(target_dir, expected_count)
        manifest_path = _write_manifest(existing, target_dir)
        return {
            "target_dir": str(target_dir),
            "source_zip": str(zip_path),
            "used_archive": "[existing_target]",
            "clip_count": len(existing),
            "manifest_path": str(manifest_path),
            "deleted_source": False,
            "actions": actions + ["reused_existing_target"],
        }

    target_dir.mkdir(parents=True, exist_ok=True)

    zip_ok, zip_detail = _zip_is_valid(zip_path)
    actions.append(f"zip_validate={zip_detail}")
    archive_to_extract = zip_path

    if not zip_ok:
        if repair_mode == "validate-only":
            raise RuntimeError(
                f"Zip validation failed ({zip_detail}) and repair_mode=validate-only. "
                "Please re-download archive."
            )
        _bsdtar_ok, bsdtar_detail = _bsdtar_can_list(zip_path)
        actions.append(f"bsdtar_list={bsdtar_detail}")

        with tempfile.TemporaryDirectory(prefix="zip_repair_", dir=str(target_parent)) as tmp:
            repaired, repair_attempts = _attempt_repair(zip_path, Path(tmp))
            actions.extend(repair_attempts)
            if repaired is None:
                raise RuntimeError(
                    "Zip repair failed. Archive appears corrupt/multipart-inconsistent. "
                    "Please re-download the dataset zip from source."
                )
            archive_to_extract = repaired

            extract_root = Path(tmp) / "extract_tmp"
            extract_root.mkdir(parents=True, exist_ok=True)
            proc_extract = _run_command(
                ["unzip", "-q", "-o", str(archive_to_extract), "-d", str(extract_root)]
            )
            if proc_extract.returncode != 0:
                raise RuntimeError(
                    "Extraction failed after repair attempt: "
                    + (proc_extract.stderr or proc_extract.stdout or "unknown unzip error")
                )
            actions.append("extract_from_repaired_archive")

            video_files = _candidate_video_files(extract_root)
            if len(video_files) != expected_count:
                raise RuntimeError(
                    f"Extracted {len(video_files)} matching clips; expected {expected_count}. "
                    "Re-download archive to avoid partial/corrupt dataset."
                )

            names = [p.name for p in video_files]
            if len(set(names)) != len(names):
                raise RuntimeError("Duplicate clip filenames found in extracted archive.")

            for src in video_files:
                shutil.move(str(src), str(target_dir / src.name))
    else:
        with tempfile.TemporaryDirectory(prefix="zip_extract_", dir=str(target_parent)) as tmp:
            extract_root = Path(tmp) / "extract_tmp"
            extract_root.mkdir(parents=True, exist_ok=True)
            proc_extract = _run_command(
                ["unzip", "-q", "-o", str(archive_to_extract), "-d", str(extract_root)]
            )
            if proc_extract.returncode != 0:
                raise RuntimeError(
                    "Extraction failed: "
                    + (proc_extract.stderr or proc_extract.stdout or "unknown unzip error")
                )
            actions.append("extract_from_original_archive")

            video_files = _candidate_video_files(extract_root)
            if len(video_files) != expected_count:
                raise RuntimeError(
                    f"Extracted {len(video_files)} matching clips; expected {expected_count}."
                )

            names = [p.name for p in video_files]
            if len(set(names)) != len(names):
                raise RuntimeError("Duplicate clip filenames found in extracted archive.")

            for src in video_files:
                shutil.move(str(src), str(target_dir / src.name))

    final_files = _verify_target_inventory(target_dir, expected_count)
    manifest_path = _write_manifest(final_files, target_dir)

    deleted_source = False
    if delete_source:
        try:
            zip_path.unlink()
            deleted_source = True
            actions.append("source_zip_deleted")
        except Exception as exc:
            actions.append(f"source_zip_delete_failed:{exc}")

    actions.append(f"manifest_written:{manifest_path.name}")

    return {
        "target_dir": str(target_dir),
        "source_zip": str(zip_path),
        "used_archive": str(archive_to_extract),
        "clip_count": len(final_files),
        "manifest_path": str(manifest_path),
        "deleted_source": deleted_source,
        "actions": actions,
    }


def validate_raw_input(input_dir: Path,
                       expected_count: int | None = None) -> dict:
    """Validate demo input clips plus optional source_map.csv provenance sidecar."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv"}
    )
    if expected_count is not None and len(files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} clips, found {len(files)} in {input_dir}")

    source_map_path = input_dir / "source_map.csv"
    if not source_map_path.exists():
        return {
            "input_dir": str(input_dir),
            "clip_count": len(files),
            "source_map_path": str(source_map_path),
            "missing_source_map_rows": [],
            "missing_clip_files": [],
            "missing_columns": [],
            "valid": True,
            "provenance_mode": "folder_inferred",
        }

    with open(source_map_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_name = {row.get("filename", "").strip(): row for row in rows if row.get("filename")}
    missing_rows = [p.name for p in files if p.name not in by_name]
    missing_files = [name for name in by_name if not (input_dir / name).exists()]

    required_columns = {"filename", "source_url", "license", "attribution", "demo_category"}
    present_columns = set(rows[0].keys()) if rows else set()
    missing_columns = sorted(required_columns - present_columns)

    return {
        "input_dir": str(input_dir),
        "clip_count": len(files),
        "source_map_path": str(source_map_path),
        "missing_source_map_rows": missing_rows,
        "missing_clip_files": missing_files,
        "missing_columns": missing_columns,
        "valid": not missing_rows and not missing_files and not missing_columns,
    }
