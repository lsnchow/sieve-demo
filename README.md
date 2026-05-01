# Sieve Egocentric QA Pipeline Demo

This repo demonstrates a public-data video curation workflow:

`raw video -> scored/indexed/reviewed/exported training dataset`

The existing video-processing modules are preserved and repurposed as a compact dataset-curation pipeline:

- `ffprobe` ingest and metadata indexing
- MediaPipe hand tracking with hand-derived egocentric features
- blur, brightness, motion, stability, and static/duplicate heuristics
- score aggregation into `training_value_score`
- manual review plus optional labeling
- dataset manifest, JSON export, SQLite index, thumbnails, and clip export

## Repository Layout

The runnable code is split by responsibility:

- `backend/` - Python pipeline source
- `src/` - React frontend
- `vite.config.ts` - local Vite runtime bridge for dataset runs and uploads during dev
- `run_dataset_demo.py` - Python CLI wrapper into `backend.cli`
- `config/default.yaml` - public thresholds and heuristic weights

The main Python CLI entrypoint is:

```bash
python run_dataset_demo.py show-config
```

## Technical Flow

```mermaid
flowchart TD
    A["Public source videos"] --> B["scripts/fetch_public_demo_inputs.py"]
    B --> C["raw_input/*.mp4"]
    B --> D["raw_input/source_map.csv"]

    C --> E["CLI: prepare-dataset"]
    D --> E
    E --> F["Flat-input validation<br/>clip count, filenames, provenance fields"]

    C --> G["CLI: run-all / autogate"]
    D --> G

    G --> H["ingest.py<br/>ffprobe metadata, hashes, source metadata merge"]
    H --> I["artifacts/<clip>/metrics.json"]

    G --> J["sampler.py<br/>keyframe extraction"]
    J --> K["artifacts/<clip>/keyframes/*.jpg"]

    G --> L["manual_precheck.py<br/>brightness, blur, jump-cut, static/duplicate heuristics"]
    L --> M["artifacts/<clip>/manual_precheck.json"]

    G --> N["stability_qa.py<br/>probe + refine camera motion analysis"]
    N --> O["artifacts/<clip>/stability_report.json"]

    G --> P["wide_lens_qa.py<br/>perspective / wide-lens fit heuristics"]
    P --> Q["artifacts/<clip>/wide_lens_report.json"]

    G --> R["hands_qa.py<br/>MediaPipe Hand Landmarker"]
    R --> S["Hand-derived features"]
    S --> S1["hand_visible_ratio"]
    S --> S2["hand_presence_score"]
    S --> S3["hand_motion_score"]
    S --> S4["hand_count_distribution"]
    S --> S5["hand_region_distribution"]
    S --> S6["egocentric_proxy_score"]
    R --> T["artifacts/<clip>/hands_report.json"]

    I --> U["gate.py + curation.py"]
    M --> U
    O --> U
    Q --> U
    T --> U

    U --> V["Public aggregation"]
    V --> V1["quality_flags"]
    V --> V2["curation_reasons"]
    V --> V3["training_value_score"]
    V --> V4["status = recommended / needs_review / low_value"]
    V --> W["artifacts/qa_metrics.csv"]

    W --> X["reviewer.py<br/>manual curation review"]
    K --> X
    M --> X
    O --> X
    Q --> X
    T --> X
    X --> Y["artifacts/<clip>/manual_review.json"]

    Y --> Z["labeler.py<br/>task/object/environment metadata"]
    Z --> AA["artifacts/<clip>/labels.json"]

    W --> AB["report.py export layer"]
    Y --> AB
    AA --> AB
    K --> AB
    C --> AB
    D --> AB

    AB --> AC["exports/dataset_manifest.csv"]
    AB --> AD["exports/dataset_manifest.json"]
    AB --> AE["exports/dataset_index.sqlite"]
    AB --> AF["exports/summary.json"]
    AB --> AG["exports/thumbnails/*.jpg"]
    AB --> AH["exports/clips/*.mp4<br/>recommended clips only"]
```

## Setup

Use Python 3.11+. The current codebase already relies on 3.10+ syntax, and the demo is validated on 3.11.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Clone the repo, create the Python environment, install dependencies, and run the pipeline on a folder of videos.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_dataset_demo.py show-config
python run_dataset_demo.py run-all -i /absolute/path/to/videos --snapshot-output-dir exports
```

If you want to test a single video file, put it in its own folder first, then point `-i` at that folder.

That produces:

- `artifacts/` - per-clip analysis artifacts
- `exports/dataset_manifest.csv`
- `exports/dataset_manifest.json`
- `exports/dataset_index.sqlite`
- `exports/summary.json`

## Demo Frontend

A lightweight Vite + React viewer is included for a 60-second read-only demo of the exported artifacts.

It reads directly from the static files in `exports/`:

- `exports/summary.json`
- `exports/dataset_manifest.json`
- `exports/thumbnails/*`
- `exports/clips/*`

Run it locally:

```bash
npm install
npm run dev
```

Build the static viewer:

```bash
npm run build
```

The frontend reads from `exports/` and uses a local Vite dev bridge for folder runs and uploads during development.

## Outputs

The public export surface is:

```text
exports/
  clips/
  thumbnails/
  dataset_manifest.csv
  dataset_manifest.json
  dataset_index.sqlite
  summary.json
```

`dataset_manifest.*` includes one row per analyzed clip with:

- `status`: `recommended`, `needs_review`, or `low_value`
- `training_value_score`
- `quality_flags`
- `curation_reasons`
- hand-derived egocentric features
- quality/stability heuristics
- provenance and `demo_category`
- optional manual labels

`dataset_index.sqlite` mirrors the manifest in a queryable form and adds indexes for `status`, `training_value_score`, `quality_flags`, `curation_reasons`, and `demo_category`.

## Heuristic Scoring

Thresholds and weights live in [`config/default.yaml`](/Users/lucas/Desktop/Sieve2/config/default.yaml). They are public heuristic defaults tuned for demo use on public or self-recorded clips.

The current score combines:

- `hand_visible_ratio`
- `hand_presence_score`
- `hand_motion_score`
- `egocentric_proxy_score`
- `brightness_score`
- `blur_score`
- `motion_score`
- `camera_stability_score`
- `static_duplicate_score`
