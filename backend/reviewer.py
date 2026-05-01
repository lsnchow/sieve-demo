"""Manual review support: keyframe contact sheets, review summaries, and curation CLI."""

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import config
from .manual_precheck import run_manual_precheck
from .prompting import get_prompter
from .utils import clip_artifacts_dir

log = logging.getLogger("pov_qa.reviewer")

MANUAL_COMBINED_ID = "content_quality"
MANUAL_COMBINED_DESC = (
    "Original content + adequate lighting + acceptable recording quality + continuous footage "
    "(no watermarks/overlays/pro editing, clearly visible, no extreme blur/compression, no jump cuts)"
)
MANUAL_POV_ID = "egocentric_fit"
MANUAL_POV_DESC = "Head-mounted POV (not chest-mounted, handheld phone, or third-person)"


def _launch_video_player(video_path: Path) -> tuple[bool, str]:
    """Open a video file in the OS default player."""
    if not video_path.exists():
        return False, "file_not_found"

    try:
        system = platform.system().lower()
        if system == "darwin":
            subprocess.Popen(["open", str(video_path)])
        elif system == "linux":
            subprocess.Popen(["xdg-open", str(video_path)])
        elif system == "windows":
            subprocess.Popen(["cmd", "/c", "start", "", str(video_path)])
        else:
            return False, f"unsupported_os_{system}"
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def generate_contact_sheet(clip_dir: Path) -> Path | None:
    """Build a grid image from keyframes in the clip's artifacts dir."""
    keyframes_dir = clip_dir / "keyframes"
    if not keyframes_dir.exists():
        return None

    kf_files = sorted(keyframes_dir.glob("*.jpg"))
    if not kf_files:
        return None

    images = [cv2.imread(str(f)) for f in kf_files]
    images = [img for img in images if img is not None]
    if not images:
        return None

    # Resize all to same dimensions
    target_h, target_w = 180, 320
    resized = [cv2.resize(img, (target_w, target_h)) for img in images]

    cols = config.CONTACT_SHEET_COLS
    rows = (len(resized) + cols - 1) // cols

    # Pad to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_imgs = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    sheet = np.vstack(grid_rows)

    # Add labels
    for i, f in enumerate(kf_files):
        row_idx = i // cols
        col_idx = i % cols
        x = col_idx * target_w + 5
        y = row_idx * target_h + 15
        label = f.stem.split("_", 1)[-1] if "_" in f.stem else f.stem
        cv2.putText(sheet, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)

    out_path = clip_dir / "review_sheet.png"
    cv2.imwrite(str(out_path), sheet)
    return out_path


def generate_review_summary(clip_dir: Path) -> Path | None:
    """Create a text summary of where to look during manual review."""
    lines = [f"=== Review Summary: {clip_dir.name} ===\n"]

    metrics_path = clip_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            meta = json.load(f)
        lines.append(f"Resolution: {meta['display_w']}x{meta['display_h']}")
        lines.append(f"FPS: {meta['effective_fps']}")
        lines.append(f"Duration: {meta['duration']:.1f}s")
        lines.append("")

    hands_path = clip_dir / "hands_report.json"
    if hands_path.exists():
        with open(hands_path) as f:
            hands = json.load(f)
        lines.append(f"Hands visible ratio: {hands['hands_ratio']:.3f}")
        if hands.get("missing_segments"):
            lines.append("Hands missing segments:")
            for seg in hands["missing_segments"]:
                lines.append(f"  {seg['start_fmt']}-{seg['end_fmt']}")
        lines.append("")

    wl_path = clip_dir / "wide_lens_report.json"
    if wl_path.exists():
        with open(wl_path) as f:
            wl = json.load(f)
        lines.append(f"Wide-lens score: {wl['wide_lens_score']:.3f} "
                      f"(confidence: {wl['confidence']:.3f})")
        lines.append("")

    stab_path = clip_dir / "stability_report.json"
    if stab_path.exists():
        with open(stab_path) as f:
            stab = json.load(f)
        lines.append(f"Max motion score: {stab['max_motion_score']:.1f}")
        if stab.get("spike_segments"):
            lines.append("Shaky segments:")
            for seg in stab["spike_segments"]:
                lines.append(
                    f"  {seg['start_fmt']}-{seg['end_fmt']} "
                    f"(score={seg['motion_score']:.1f})"
                )
        lines.append("")

    precheck_path = clip_dir / "manual_precheck.json"
    if precheck_path.exists():
        with open(precheck_path) as f:
            pre = json.load(f)
        lines.append("--- Manual Precheck Flags (assistive only) ---")
        lines.append(
            f"Dark ratio: {pre.get('dark_ratio', 0):.3f} | "
            f"Blurry ratio: {pre.get('blurry_ratio', 0):.3f} | "
            f"Max cut score: {pre.get('max_cut_score', 0):.3f}"
        )
        cuts = pre.get("candidate_jump_cuts", [])
        if cuts:
            lines.append("Possible jump cuts (inspect manually):")
            for c in cuts[:10]:
                lines.append(f"  ~{c['time_fmt']} (score={c['score']:.3f})")
        dark = pre.get("dark_segments", [])
        if dark:
            lines.append("Dark segments:")
            for seg in dark[:6]:
                lines.append(f"  {seg['start_fmt']}-{seg['end_fmt']}")
        blur = pre.get("blurry_segments", [])
        if blur:
            lines.append("Blurry segments:")
            for seg in blur[:6]:
                lines.append(f"  {seg['start_fmt']}-{seg['end_fmt']}")
        lines.append("")

    lines.append("--- Manual Review Checklist ---")
    lines.append(f"[ ] {MANUAL_COMBINED_ID}: {MANUAL_COMBINED_DESC}")
    lines.append(f"[ ] {MANUAL_POV_ID}: {MANUAL_POV_DESC}")

    out_path = clip_dir / "review_summary.txt"
    out_path.write_text("\n".join(lines))
    return out_path


def generate_all_review_materials(artifacts_dir: Path | None = None,
                                  qa_metrics_path: Path | None = None,
                                  input_dir: Path | None = None,
                                  run_precheck: bool = False):
    """Generate contact sheets and review summaries for clips needing review."""
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR

    if qa_metrics_path is None:
        qa_metrics_path = artifacts_dir / "qa_metrics.csv"

    if not qa_metrics_path.exists():
        log.error("qa_metrics.csv not found at %s — run autogate first", qa_metrics_path)
        return

    df = pd.read_csv(qa_metrics_path)
    review_clips = df[df["gate_result"].isin([config.GATE_PASS, config.GATE_BORDERLINE])]

    log.info("Generating review materials for %d clips", len(review_clips))
    for _, row in review_clips.iterrows():
        stem = Path(row["filename"]).stem
        clip_dir = artifacts_dir / stem
        if clip_dir.exists():
            # Heuristic assist only; does not auto-reject.
            video_path = None
            if input_dir:
                candidate = Path(input_dir) / row["filename"]
                if candidate.exists():
                    video_path = candidate
            if run_precheck and video_path is not None:
                run_manual_precheck(video_path, artifacts_dir=artifacts_dir)
            generate_contact_sheet(clip_dir)
            generate_review_summary(clip_dir)
            log.info("  %s: review materials generated", row["filename"])


def interactive_review(artifacts_dir: Path | None = None,
                       qa_metrics_path: Path | None = None,
                       input_dir: Path | None = None,
                       autoplay: bool = True,
                       on_decision_saved: Callable[[dict], bool] | None = None) -> list[dict]:
    """Interactive CLI for manual curation review.

    Returns list of review decision dicts.
    """
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR

    if qa_metrics_path is None:
        qa_metrics_path = artifacts_dir / "qa_metrics.csv"

    if not qa_metrics_path.exists():
        log.error("qa_metrics.csv not found — run autogate first")
        return []

    df = pd.read_csv(qa_metrics_path)
    review_clips = df[df["gate_result"].isin([config.GATE_PASS, config.GATE_BORDERLINE])]

    if review_clips.empty:
        print("No clips to review — all clips were routed to low_value automatically.")
        return []

    decisions = []
    existing = _load_existing_reviews(artifacts_dir)
    prompter = get_prompter()

    for idx, (_, row) in enumerate(review_clips.iterrows()):
        filename = row["filename"]
        stem = Path(filename).stem

        if stem in existing:
            print(f"\n[{idx+1}/{len(review_clips)}] {filename} — already reviewed, skipping")
            decisions.append(existing[stem])
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(review_clips)}] {filename}")
        print(f"Gate: {row['gate_result']}")
        print(f"Resolution: {row['display_w']}x{row['display_h']} | "
              f"FPS: {row['effective_fps']} | Duration: {row['duration']:.1f}s")

        if pd.notna(row.get("c3a_hands_ratio")):
            print(f"Hands ratio: {row['c3a_hands_ratio']:.3f} | "
                  f"Wide-lens: {row.get('c3b_wide_score', 'N/A')}")
        if pd.notna(row.get("c9_max_motion")):
            print(f"Max motion: {row['c9_max_motion']:.1f}")

        clip_dir = artifacts_dir / stem
        summary_path = clip_dir / "review_summary.txt"
        if summary_path.exists():
            print(f"\nReview summary: {summary_path}")
        sheet_path = clip_dir / "review_sheet.png"
        if sheet_path.exists():
            print(f"Contact sheet:  {sheet_path}")

        action = prompter.select(
            "Clip action",
            [
                ("Review this clip", "review"),
                ("Skip for now", "skip"),
                ("Quit review session", "quit"),
            ],
            default="review",
        )
        if action == "skip":
            continue
        if action == "quit":
            break

        if autoplay and input_dir is not None:
            video_path = input_dir / filename
            ok, detail = _launch_video_player(video_path)
            if ok:
                print(f"Opened video player: {video_path}")
            else:
                print(f"Could not auto-open video ({detail}): {video_path}")

        print(f"\n--- Manual Criteria ---")
        while True:
            review = {"filename": filename, "approved": True, "reject_reasons": []}

            combined_choice = prompter.select(
                f"{MANUAL_COMBINED_ID}: {MANUAL_COMBINED_DESC}",
                [
                    ("Pass", "pass"),
                    ("Mark low-value", "reject"),
                    ("Skip (leave undecided)", "skip"),
                ],
                default="pass",
            )
            if combined_choice == "reject":
                failed_ids = prompter.text(
                    f"Which parts of {MANUAL_COMBINED_ID} failed? (comma-separated)",
                    default=MANUAL_COMBINED_ID,
                    allow_empty=False,
                )
                reason = prompter.text(
                    f"Low-value reason for {MANUAL_COMBINED_ID}",
                    allow_empty=False,
                )
                ts = prompter.text(
                    "Timestamp(s) [optional]",
                    allow_empty=True,
                )
                full_reason = f"{failed_ids}: {reason}"
                if ts:
                    full_reason += f" at {ts}"
                review["reject_reasons"].append(full_reason)
                review["approved"] = False

            # Short-circuit: only ask the egocentric-fit question if content quality passed.
            if review["approved"]:
                c8_choice = prompter.select(
                    f"{MANUAL_POV_ID}: {MANUAL_POV_DESC}",
                    [
                        ("Pass", "pass"),
                        ("Mark low-value", "reject"),
                        ("Skip (leave undecided)", "skip"),
                    ],
                    default="pass",
                )
                if c8_choice == "reject":
                    reason = prompter.text(
                        f"Low-value reason for {MANUAL_POV_ID}",
                        allow_empty=False,
                    )
                    ts = prompter.text(
                        "Timestamp(s) [optional]",
                        allow_empty=True,
                    )
                    full_reason = f"{MANUAL_POV_ID}: {reason}"
                    if ts:
                        full_reason += f" at {ts}"
                    review["reject_reasons"].append(full_reason)
                    review["approved"] = False

            # For borderline clips, resolve uncertain automated checks only if
            # manual criteria did not already reject the clip.
            if row["gate_result"] == config.GATE_BORDERLINE and review["approved"]:
                print("\nThis clip needs manual review. Resolve the uncertain automated signals:")
                borderline_items = []
                if row.get("c3a_gate") == "borderline":
                    borderline_items.append(("Hands visibility", row.get("c3a_detail", "")))
                if row.get("c3b_gate") == "borderline":
                    borderline_items.append(("Perspective fit", row.get("c3b_detail", "")))
                if row.get("c9_gate") == "borderline":
                    borderline_items.append(("Camera stability", row.get("c9_detail", "")))

                for item_name, detail in borderline_items:
                    choice = prompter.select(
                        f"{item_name}: {detail}",
                        [
                            ("Mark PASS", "pass"),
                            ("Mark low-value on this signal", "reject"),
                        ],
                        default="pass",
                    )
                    if choice == "reject":
                        note = prompter.text(
                            f"Manual low-value note for {item_name}",
                            allow_empty=True,
                        )
                        ts = prompter.text(
                            "Timestamp(s) [optional]",
                            allow_empty=True,
                        )
                        full_reason = detail
                        if note:
                            full_reason = f"{full_reason}; {note}"
                        if ts:
                            full_reason = f"{full_reason} at {ts}"
                        review["reject_reasons"].append(full_reason)
                        review["approved"] = False
                        # Short-circuit on first borderline automated rejection.
                        break

            status = "RECOMMENDED" if review["approved"] else "LOW_VALUE"
            print(f"\n=> {status}")
            if review["reject_reasons"]:
                print(f"Reasons: {'; '.join(review['reject_reasons'])}")

            post_action = prompter.select(
                "Decision action",
                [
                    ("Save and continue", "save"),
                    ("Redo this clip", "redo"),
                    ("Skip without saving", "skip"),
                    ("Quit review session", "quit"),
                ],
                default="save",
            )
            if post_action == "redo":
                continue
            if post_action == "skip":
                review = None
                break
            if post_action == "quit":
                if review is not None:
                    review_path = clip_dir / "manual_review.json"
                    with open(review_path, "w") as f:
                        json.dump(review, f, indent=2)
                    decisions.append(review)
                    if on_decision_saved is not None:
                        should_continue = on_decision_saved(review)
                        if should_continue is False:
                            return decisions
                return decisions

            # Save per-clip review
            review_path = clip_dir / "manual_review.json"
            with open(review_path, "w") as f:
                json.dump(review, f, indent=2)
            decisions.append(review)
            if on_decision_saved is not None:
                should_continue = on_decision_saved(review)
                if should_continue is False:
                    return decisions
            break

    return decisions


def build_vetting_dashboard(artifacts_dir: Path | None = None,
                            qa_metrics_path: Path | None = None) -> pd.DataFrame:
    """Build a dashboard dataframe spanning automated + manual + labeling status."""
    artifacts_dir = artifacts_dir or config.ARTIFACTS_DIR
    if qa_metrics_path is None:
        qa_metrics_path = artifacts_dir / "qa_metrics.csv"
    if not qa_metrics_path.exists():
        return pd.DataFrame()

    qa_df = pd.read_csv(qa_metrics_path).sort_values("filename").reset_index(drop=True)
    rows = []

    for _, row in qa_df.iterrows():
        filename = row["filename"]
        stem = Path(filename).stem
        clip_dir = artifacts_dir / stem

        review_path = clip_dir / "manual_review.json"
        labels_path = clip_dir / "labels.json"

        manual_status = "pending"
        manual_reason = ""
        if review_path.exists():
            with open(review_path) as f:
                review = json.load(f)
            if review.get("approved", False):
                manual_status = "recommended"
            else:
                manual_status = "low_value"
                manual_reason = "; ".join(review.get("reject_reasons", []))

        label_status = "n/a"
        if manual_status == "recommended":
            label_status = "done" if labels_path.exists() else "pending"

        gate_result = row.get("gate_result", "")
        auto_reason = row.get("reject_reasons", "")
        if pd.isna(auto_reason):
            auto_reason = ""

        if gate_result == config.GATE_FAIL:
            final_status = config.PUBLIC_STATUS_LOW_VALUE
            next_step = "-"
            reason = auto_reason
        elif manual_status == "pending":
            final_status = config.PUBLIC_STATUS_NEEDS_REVIEW
            next_step = "review"
            reason = auto_reason
        elif manual_status == "low_value":
            final_status = config.PUBLIC_STATUS_LOW_VALUE
            next_step = "-"
            reason = "; ".join(x for x in [auto_reason, manual_reason] if x)
        elif label_status == "pending":
            final_status = config.PUBLIC_STATUS_RECOMMENDED
            next_step = "label"
            reason = ""
        else:
            final_status = config.PUBLIC_STATUS_RECOMMENDED
            next_step = "-"
            reason = ""

        rows.append({
            "filename": filename,
            "gate_result": gate_result,
            "manual_status": manual_status,
            "label_status": label_status,
            "final_status": final_status,
            "next_step": next_step,
            "reason": reason,
        })

    return pd.DataFrame(rows)


def print_vetting_dashboard(df: pd.DataFrame) -> None:
    """Pretty-print dashboard summary for CLI use."""
    if df.empty:
        print("No qa_metrics.csv found. Run autogate first.")
        return

    print(f"\n{'='*76}")
    print("Dataset Curation Dashboard")
    print(f"{'='*76}")
    print(f"Total clips: {len(df)}")
    final_counts = df["final_status"].value_counts().to_dict()
    print(
        "Final status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(final_counts.items()))
    )
    print(
        "Action queue: "
        f"review={int((df['next_step'] == 'review').sum())}, "
        f"label={int((df['next_step'] == 'label').sum())}"
    )

    display_cols = [
        "filename", "gate_result", "manual_status",
        "label_status", "final_status", "next_step",
    ]
    print("")
    print(df[display_cols].to_string(index=False))


def _load_existing_reviews(artifacts_dir: Path) -> dict:
    """Load previously completed manual reviews."""
    existing = {}
    for d in artifacts_dir.iterdir():
        if d.is_dir():
            review_file = d / "manual_review.json"
            if review_file.exists():
                with open(review_file) as f:
                    review = json.load(f)
                existing[d.name] = review
    return existing
