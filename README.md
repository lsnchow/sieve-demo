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

The MediaPipe hand model downloads automatically on first hand-analysis run if `models/hand_landmarker.task` is missing.

## Fetch Public Demo Inputs

The demo input set is a flat `raw_input/` directory plus `raw_input/source_map.csv`.

```bash
.venv/bin/python scripts/fetch_public_demo_inputs.py
```

The fetch script pulls public videos from Wikimedia Commons, cuts them into shorter demo clips, and writes clip provenance and `demo_category` metadata into the sidecar CSV.

## Run The Demo

```bash
bash demo.sh
```

Or step-by-step:

```bash
.venv/bin/python run_dataset_demo.py show-config
.venv/bin/python run_dataset_demo.py prepare-dataset --input-dir raw_input --expected-count 10
.venv/bin/python run_dataset_demo.py run-all -i raw_input -n 10 --snapshot-output-dir exports
.venv/bin/python run_dataset_demo.py export -o exports -n 10
```

Manual review and labeling remain available when you want to override or enrich the automated results:

```bash
.venv/bin/python run_dataset_demo.py review --input-dir raw_input --output-dir exports
.venv/bin/python run_dataset_demo.py label --output-dir exports
```

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

Thresholds and weights live in [`config/default.yaml`](/Users/lucas/Desktop/Sieve2/config/default.yaml). They are public heuristic defaults tuned for demo use on public or self-recorded clips, not confidential rubric values.

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

## Notes

- MediaPipe is the required hand-tracking backend in this demo.
- The existing manual labeling surface is kept as the action/object metadata layer; no new action-recognition model is introduced.
- The pipeline still emits the original internal artifacts under `artifacts/`, but the public-facing demo surface is the `exports/` directory.
