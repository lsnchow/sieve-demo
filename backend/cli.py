"""CLI entry point for the public video dataset-curation demo."""

import json
import logging
from pathlib import Path

import click

from . import config
from .utils import collect_video_files, configure_runtime_threads, setup_logging


def _print_output_paths(output_dir: Path) -> None:
    """Print canonical deliverable paths for current session."""
    click.echo("\nDataset exports:")
    click.echo(f"  {output_dir / 'dataset_manifest.csv'}")
    click.echo(f"  {output_dir / 'dataset_manifest.json'}")
    click.echo(f"  {output_dir / 'summary.json'}")
    click.echo(f"  {output_dir / 'dataset_index.sqlite'}")


def _apply_threshold_overrides(
    c9_spike_threshold: float | None,
    c9_min_fail_segment_sec: float | None,
    wide_pass_threshold: float | None,
    wide_borderline_low: float | None,
) -> None:
    log = logging.getLogger("pov_qa.cli")
    if c9_spike_threshold is not None:
        config.STABILITY_SPIKE_THRESHOLD = float(c9_spike_threshold)
        log.info("Override: STABILITY_SPIKE_THRESHOLD=%.2f", config.STABILITY_SPIKE_THRESHOLD)
    if c9_min_fail_segment_sec is not None:
        config.STABILITY_MIN_FAIL_SEGMENT_SEC = float(c9_min_fail_segment_sec)
        log.info(
            "Override: STABILITY_MIN_FAIL_SEGMENT_SEC=%.2f",
            config.STABILITY_MIN_FAIL_SEGMENT_SEC,
        )
    if wide_pass_threshold is not None:
        config.WIDE_LENS_PASS_THRESHOLD = float(wide_pass_threshold)
        log.info("Override: WIDE_LENS_PASS_THRESHOLD=%.2f", config.WIDE_LENS_PASS_THRESHOLD)
    if wide_borderline_low is not None:
        config.WIDE_LENS_BORDERLINE_LOW = float(wide_borderline_low)
        log.info("Override: WIDE_LENS_BORDERLINE_LOW=%.2f", config.WIDE_LENS_BORDERLINE_LOW)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--artifacts-dir", type=click.Path(), default="artifacts",
              help="Base directory for per-clip artifacts")
@click.option("--config-path", type=click.Path(), default=str(config.DEFAULT_CONFIG_PATH),
              show_default=True, help="Path to YAML config overrides")
def cli(verbose, artifacts_dir, config_path):
    """Raw video to scored, indexed, reviewed, exported training dataset."""
    setup_logging(verbose)
    configure_runtime_threads()
    config.load_yaml_config(Path(config_path))
    config.ARTIFACTS_DIR = Path(artifacts_dir)


@cli.command("show-config")
def show_config():
    """Print active threshold/runtime config used by the pipeline."""
    payload = config.get_public_config()
    click.echo(json.dumps(payload, indent=2))


@cli.command()
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True),
              help="Directory containing video clips")
@click.option("--expected-count", "-n", type=int, default=None,
              help="Expected number of clips (asserts count matches)")
def ingest(input_dir, expected_count):
    """Ingest flat source clips, ffprobe metadata, and provenance sidecar fields."""
    from .ingest import run_ingest
    results = run_ingest(Path(input_dir), expected_count)
    click.echo(f"\nIngested {len(results)} clips.")

    passed = sum(1 for r in results if r["c1_pass"] and r["c2_pass"])
    click.echo(f"Baseline format checks passed: {passed}/{len(results)}")


@cli.command()
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True),
              help="Directory containing video clips")
@click.option("--expected-count", "-n", type=int, default=None,
              help="Expected number of clips")
@click.option("--c9-spike-threshold", type=float, default=None,
              help="Override stability spike threshold (higher = more lenient)")
@click.option("--c9-min-fail-segment-sec", type=float, default=None,
              help="Override minimum shaky duration for stability hard fail")
@click.option("--wide-pass-threshold", type=float, default=None,
              help="Override wide-lens pass threshold (higher = stricter)")
@click.option("--wide-borderline-low", type=float, default=None,
              help="Override wide-lens hard-fail threshold")
def autogate(input_dir, expected_count, c9_spike_threshold,
             c9_min_fail_segment_sec, wide_pass_threshold, wide_borderline_low):
    """Run automated analysis and generate public curation scores."""
    from .gate import run_autogate
    _apply_threshold_overrides(
        c9_spike_threshold,
        c9_min_fail_segment_sec,
        wide_pass_threshold,
        wide_borderline_low,
    )
    df = run_autogate(Path(input_dir), expected_count)
    click.echo(f"\nAutogate complete. Results in {config.ARTIFACTS_DIR / 'qa_metrics.csv'}")


@cli.command()
@click.option("--input-dir", "-i", type=click.Path(exists=True), default=None,
              help="Directory containing video clips (for reference)")
@click.option("--run-precheck/--no-run-precheck", default=False,
              help="Run heuristic precheck before manual review (slower)")
@click.option("--autoplay/--no-autoplay", default=True,
              help="Auto-open each clip in your system video player during review")
@click.option("--output-dir", "-o", type=click.Path(), default=str(config.EXPORT_DIR),
              show_default=True, help="Directory for live dataset exports")
@click.option("--live-export/--no-live-export", default=True,
              help="Rebuild dataset exports after each saved decision")
def review(input_dir, run_precheck, autoplay, output_dir, live_export):
    """Interactive manual review for clips that need curation decisions."""
    from .report import export_live_snapshot
    from .reviewer import generate_all_review_materials, interactive_review
    from .vetting_state import update_and_write_vetting_state

    output_dir = Path(output_dir)
    click.echo(f"\nUsing QA source: {config.ARTIFACTS_DIR / 'qa_metrics.csv'}")
    _print_output_paths(output_dir)

    generate_all_review_materials(
        input_dir=Path(input_dir) if input_dir else None,
        run_precheck=run_precheck,
    )
    if live_export:
        export_live_snapshot(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
        )
        update_and_write_vetting_state(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
        )

    def _on_decision_saved(review: dict) -> bool:
        if live_export:
            export_live_snapshot(
                artifacts_dir=config.ARTIFACTS_DIR,
                output_dir=output_dir,
            )
        update_and_write_vetting_state(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
            last_clip=review.get("filename"),
        )
        return True

    decisions = interactive_review(
        input_dir=Path(input_dir) if input_dir else None,
        autoplay=autoplay,
        on_decision_saved=_on_decision_saved,
    )

    recommended = sum(1 for d in decisions if d.get("approved"))
    low_value = len(decisions) - recommended
    click.echo(f"\nReview complete: {recommended} recommended, {low_value} low-value")


@cli.command("vet-dashboard")
@click.option("--qa-metrics-path", type=click.Path(exists=True), default=None,
              help="Optional qa_metrics.csv path (defaults to artifacts/qa_metrics.csv)")
def vet_dashboard(qa_metrics_path):
    """Show one-screen dashboard for automated analysis, review, and labeling."""
    from .reviewer import build_vetting_dashboard, print_vetting_dashboard

    df = build_vetting_dashboard(
        artifacts_dir=config.ARTIFACTS_DIR,
        qa_metrics_path=Path(qa_metrics_path) if qa_metrics_path else None,
    )
    print_vetting_dashboard(df)


@cli.command("vet")
@click.option("--input-dir", "-i", type=click.Path(exists=True), default=None,
              help="Directory containing video clips (for precheck/review context)")
@click.option("--run-precheck/--no-run-precheck", default=False,
              help="Run heuristic precheck before manual review")
@click.option("--do-label/--no-do-label", default=True,
              help="Run labeling immediately after manual review")
@click.option("--output-dir", "-o", type=click.Path(), default=str(config.EXPORT_DIR),
              show_default=True, help="Directory for live dataset exports")
@click.option("--live-export/--no-live-export", default=True,
              help="Rebuild dataset exports after each saved decision or label")
def vet(input_dir, run_precheck, do_label, output_dir, live_export):
    """Run dashboard, manual review, and labeling in one curation workflow."""
    from .labeler import interactive_labeling, prompt_label_for_clip
    from .report import export_live_snapshot
    from .reviewer import (
        build_vetting_dashboard,
        generate_all_review_materials,
        interactive_review,
        print_vetting_dashboard,
    )
    from .vetting_state import update_and_write_vetting_state

    output_dir = Path(output_dir)
    click.echo(f"\nUsing QA source: {config.ARTIFACTS_DIR / 'qa_metrics.csv'}")
    _print_output_paths(output_dir)

    click.echo("\nCurrent status:")
    print_vetting_dashboard(build_vetting_dashboard(config.ARTIFACTS_DIR))

    if live_export:
        export_live_snapshot(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
        )
    update_and_write_vetting_state(
        artifacts_dir=config.ARTIFACTS_DIR,
        output_dir=output_dir,
    )

    def _on_label_saved(label: dict) -> bool:
        if live_export:
            export_live_snapshot(
                artifacts_dir=config.ARTIFACTS_DIR,
                output_dir=output_dir,
            )
        update_and_write_vetting_state(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
            last_clip=label.get("filename"),
        )
        return True

    # Resume behavior: process approved-but-unlabeled backlog first.
    if do_label:
        backlog_labels = interactive_labeling(
            on_label_saved=_on_label_saved,
        )
        click.echo(f"Backlog labeling complete: {len(backlog_labels)} clips")

    generate_all_review_materials(
        input_dir=Path(input_dir) if input_dir else None,
        run_precheck=run_precheck,
    )

    def _on_decision_saved(review: dict) -> bool:
        if live_export:
            export_live_snapshot(
                artifacts_dir=config.ARTIFACTS_DIR,
                output_dir=output_dir,
            )
        update_and_write_vetting_state(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
            last_clip=review.get("filename"),
        )

        if do_label and review.get("approved"):
            label_result = prompt_label_for_clip(
                review["filename"],
                artifacts_dir=config.ARTIFACTS_DIR,
                allow_skip=True,
                allow_quit=True,
                on_label_saved=_on_label_saved,
            )
            if label_result["action"] == "quit":
                return False
            # Keep state synchronized even when user skips immediate labeling.
            update_and_write_vetting_state(
                artifacts_dir=config.ARTIFACTS_DIR,
                output_dir=output_dir,
                last_clip=review.get("filename"),
            )
        return True

    decisions = interactive_review(
        input_dir=Path(input_dir) if input_dir else None,
        autoplay=True,
        on_decision_saved=_on_decision_saved,
    )
    recommended = sum(1 for d in decisions if d.get("approved"))
    low_value = len(decisions) - recommended
    click.echo(f"\nManual review complete: {recommended} recommended, {low_value} low-value")

    if live_export:
        export_live_snapshot(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
        )
    update_and_write_vetting_state(
        artifacts_dir=config.ARTIFACTS_DIR,
        output_dir=output_dir,
    )

    click.echo("\nUpdated status:")
    print_vetting_dashboard(build_vetting_dashboard(config.ARTIFACTS_DIR))


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default=str(config.EXPORT_DIR),
              show_default=True, help="Directory for live dataset exports")
@click.option("--live-export/--no-live-export", default=True,
              help="Rebuild dataset exports after each saved label")
def label(output_dir, live_export):
    """Interactive labeling for recommended clips."""
    from .report import export_live_snapshot
    from .vetting_state import update_and_write_vetting_state
    from .labeler import interactive_labeling

    output_dir = Path(output_dir)
    click.echo(f"\nUsing QA source: {config.ARTIFACTS_DIR / 'qa_metrics.csv'}")
    _print_output_paths(output_dir)

    def _on_label_saved(label: dict) -> bool:
        if live_export:
            export_live_snapshot(
                artifacts_dir=config.ARTIFACTS_DIR,
                output_dir=output_dir,
            )
        update_and_write_vetting_state(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=output_dir,
            last_clip=label.get("filename"),
        )
        return True

    labels = interactive_labeling(on_label_saved=_on_label_saved)
    click.echo(f"\nLabeled {len(labels)} clips.")


@cli.command("prepare-dataset")
@click.option("--input-dir", type=click.Path(exists=True),
              default=str(config.RAW_INPUT_DIR), show_default=True,
              help="Flat directory of source clips to validate")
@click.option("--expected-count", type=int, default=None,
              help="Optional expected number of clips")
def prepare_dataset(input_dir, expected_count):
    """Validate the flat raw-input demo set and its provenance sidecar."""
    from .dataset_prep import validate_raw_input

    result = validate_raw_input(Path(input_dir), expected_count=expected_count)
    click.echo("\nDemo input validation complete:")
    click.echo(json.dumps(result, indent=2))


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default=str(config.EXPORT_DIR),
              show_default=True, help="Directory for final dataset exports")
@click.option("--expected-total", "-n", type=int, default=None,
              help="Expected total clip count for validation")
def export(output_dir, expected_total):
    """Generate the dataset manifest, JSON export, SQLite index, and summary."""
    from .report import export_deliverables
    out_dir = Path(output_dir)
    _print_output_paths(out_dir)
    export_deliverables(output_dir=out_dir, expected_total=expected_total)


@cli.command("run-all")
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True),
              help="Directory containing video clips")
@click.option("--expected-count", "-n", type=int, default=None,
              help="Expected number of clips")
@click.option("--c9-spike-threshold", type=float, default=None,
              help="Override stability spike threshold (higher = more lenient)")
@click.option("--c9-min-fail-segment-sec", type=float, default=None,
              help="Override minimum shaky duration for stability hard fail")
@click.option("--wide-pass-threshold", type=float, default=None,
              help="Override wide-lens pass threshold (higher = stricter)")
@click.option("--wide-borderline-low", type=float, default=None,
              help="Override wide-lens hard-fail threshold")
@click.option("--timing-summary/--no-timing-summary", default=True,
              help="Print per-stage timing summary after run-all")
@click.option("--export-snapshot/--no-export-snapshot", default=True,
              help="Export dataset manifest/index snapshot after run-all")
@click.option("--snapshot-output-dir", type=click.Path(), default=None,
              help="Snapshot output dir (defaults to exports/ when omitted)")
def run_all(input_dir, expected_count, c9_spike_threshold,
            c9_min_fail_segment_sec, wide_pass_threshold, wide_borderline_low,
            timing_summary, export_snapshot, snapshot_output_dir):
    """Run ingest + autogate end-to-end (automated phases only)."""
    from .gate import run_autogate
    from .reviewer import generate_all_review_materials

    log = logging.getLogger("pov_qa.cli")
    log.info("Running full automated pipeline...")
    _apply_threshold_overrides(
        c9_spike_threshold,
        c9_min_fail_segment_sec,
        wide_pass_threshold,
        wide_borderline_low,
    )

    df = run_autogate(Path(input_dir), expected_count)

    log.info("Generating review materials...")
    generate_all_review_materials(input_dir=Path(input_dir), run_precheck=False)

    # Print summary
    total = len(df)
    pass_count = len(df[df["gate_result"] == config.GATE_PASS])
    fail_count = len(df[df["gate_result"] == config.GATE_FAIL])
    border_count = len(df[df["gate_result"] == config.GATE_BORDERLINE])

    click.echo(f"\n{'='*50}")
    click.echo("  Automated Analysis Summary")
    click.echo(f"  Total clips:       {total}")
    click.echo(f"  Auto pass:         {pass_count}")
    click.echo(f"  Auto low-value:    {fail_count}")
    click.echo(f"  Needs review:      {border_count}")
    click.echo(f"{'='*50}")
    click.echo(f"\nResults: {config.ARTIFACTS_DIR / 'qa_metrics.csv'}")
    if timing_summary:
        _print_timing_summary(df)
    if export_snapshot:
        from .report import export_deliverables

        if snapshot_output_dir:
            out_dir = Path(snapshot_output_dir)
        else:
            out_dir = config.EXPORT_DIR
        _print_output_paths(out_dir)
        export_deliverables(
            artifacts_dir=config.ARTIFACTS_DIR,
            output_dir=out_dir,
            expected_total=expected_count,
        )
        click.echo(f"Snapshot export: {out_dir}")
    click.echo("Next step: run manual review for clips marked needs_review")


@cli.command("c9-sweep")
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True),
              help="Directory containing video clips")
@click.option("--thresholds", default="30,40,50,60,70,80,90,100,110,120,130,150",
              help="Comma-separated stability thresholds for A/B sweep")
@click.option("--min-longest-spike-sec", type=float, default=0.0,
              help="Only count a reject if longest shaky segment >= this value")
@click.option("--min-total-shaky-sec", type=float, default=10.0,
              help="Only count a reject if cumulative shaky duration >= this value")
@click.option("--score-col", type=click.Choice(["max_motion_score", "peak_motion_score"]),
              default="max_motion_score",
              help="Metric used for thresholding")
@click.option("--focus-file", default=None,
              help="Optional filename to highlight (e.g. 1d7c2b3c_file1.mp4)")
@click.option("--force", is_flag=True,
              help="Recompute stability reports even if cached reports exist")
def c9_sweep(input_dir, thresholds, min_longest_spike_sec, min_total_shaky_sec,
             score_col, focus_file, force):
    """Run stability-only pass and print threshold A/B analysis."""
    from .c9_tuning import build_c9_metrics, build_threshold_sweep

    threshold_vals = []
    for t in thresholds.split(","):
        t = t.strip()
        if not t:
            continue
        threshold_vals.append(float(t))

    if not threshold_vals:
        raise click.ClickException("No valid thresholds provided")

    metrics = build_c9_metrics(Path(input_dir), config.ARTIFACTS_DIR, force=force)
    sweep = build_threshold_sweep(
        metrics,
        thresholds=threshold_vals,
        min_longest_spike_sec=min_longest_spike_sec,
        min_total_shaky_sec=min_total_shaky_sec,
        score_col=score_col,
    )

    metrics_path = config.ARTIFACTS_DIR / "c9_metrics.csv"
    sweep_path = config.ARTIFACTS_DIR / "c9_threshold_sweep.csv"
    metrics.to_csv(metrics_path, index=False)
    sweep.to_csv(sweep_path, index=False)

    click.echo("\nStability metrics (sorted by severity):")
    show_cols = ["filename", "max_motion_score", "peak_motion_score",
                 "longest_spike_sec", "report_total_shaky_sec",
                 "total_spike_sec", "spike_segments"]
    click.echo(metrics[show_cols].to_string(index=False))

    click.echo(
        f"\nThreshold A/B ({score_col}, "
        f"min_longest_spike_sec={min_longest_spike_sec}, "
        f"min_total_shaky_sec={min_total_shaky_sec}):"
    )
    for _, row in sweep.iterrows():
        click.echo(
            f"  T={row['threshold']:.1f}: low_value={int(row['rejected_count'])}/{len(metrics)}"
        )

    if focus_file:
        focus = metrics[metrics["filename"] == focus_file]
        if focus.empty:
            click.echo(f"\nFocus file not found in metrics: {focus_file}")
        else:
            focus_row = focus.iloc[0]
            fail_thresholds = []
            for t in threshold_vals:
                fails = (
                    focus_row[score_col] > t
                    and (
                        focus_row["longest_spike_sec"] >= min_longest_spike_sec
                        or focus_row["report_total_shaky_sec"] >= min_total_shaky_sec
                    )
                )
                if fails:
                    fail_thresholds.append(f"{t:g}")
            click.echo(
                f"\nFocus: {focus_file} | {score_col}={focus_row[score_col]:.2f} "
                f"| longest_spike_sec={focus_row['longest_spike_sec']:.2f} "
                f"| total_shaky_sec={focus_row['report_total_shaky_sec']:.2f}"
            )
            click.echo(
                f"Fails thresholds: {', '.join(fail_thresholds) if fail_thresholds else '[none]'}"
            )

    click.echo(f"\nSaved: {metrics_path}")
    click.echo(f"Saved: {sweep_path}")


@cli.command("manual-precheck")
@click.option("--input-dir", "-i", required=True, type=click.Path(exists=True),
              help="Directory containing video clips")
def manual_precheck(input_dir):
    """Run heuristic quality prechecks to accelerate manual review."""
    from .manual_precheck import run_manual_precheck

    files = collect_video_files(Path(input_dir))

    for fp in files:
        run_manual_precheck(fp, artifacts_dir=config.ARTIFACTS_DIR)

    click.echo(f"Manual precheck done for {len(files)} clips in {config.ARTIFACTS_DIR}")


def main():
    cli()


def _print_timing_summary(df):
    cols = ["t_widelens_sec", "t_stability_sec", "t_hands_sec", "t_total_clip_sec"]
    if any(c not in df.columns for c in cols):
        return
    timing_df = df[cols].fillna(0.0)
    click.echo("\nTiming summary (seconds):")
    for c in cols:
        click.echo(
            f"  {c}: mean={timing_df[c].mean():.3f} "
            f"p95={timing_df[c].quantile(0.95):.3f} sum={timing_df[c].sum():.3f}"
        )
    timing_json = config.ARTIFACTS_DIR / "timing_summary.json"
    if timing_json.exists():
        click.echo(f"Timing artifact: {timing_json}")


if __name__ == "__main__":
    main()
