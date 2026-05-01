# Sieve Egocentric QA Pipeline Demo: Technical Deep Dive

## Executive Summary

This repo is a compact, local-first video dataset curation system built around an existing egocentric QA pipeline. The core design principle is preservation rather than reinvention: the original working modules for video probing, frame sampling, MediaPipe hand analysis, wide-lens / perspective heuristics, stability analysis, manual review, and export were retained and repurposed into a public-data dataset curation demo.

Code layout is intentionally split into:

- `backend/` for the Python pipeline
- `src/` for the React visualization layer
- `vite.config.ts` for the local dev-only runtime bridge
- `run_dataset_demo.py` as the Python CLI entrypoint

The system turns a flat folder of public video clips into:

- per-clip analysis artifacts
- a public `training_value_score`
- public status labels: `recommended`, `needs_review`, `low_value`
- a CSV + JSON manifest
- a queryable SQLite index
- thumbnails
- a curated export set of recommended clips

The key technical decision is that this is not a learned ranking system. It is a deterministic, heuristic, auditable pipeline. Every score in the final manifest can be traced back to explicit image-processing, motion-analysis, and hand-analysis steps.

---

## 1. System Design Philosophy

The pipeline is deliberately optimized for:

1. End-to-end local reproducibility
2. Human-auditable scoring logic
3. Minimal infrastructure
4. Configurable thresholds and weights
5. Preservation of existing working code paths

This means the implementation favors:

- `ffprobe` over heavyweight media services
- OpenCV heuristics over external inference APIs
- MediaPipe for hand tracking as the required egocentric signal backbone
- YAML-configured scoring over hardcoded opaque constants
- SQLite for indexing instead of introducing a separate service

The result is a demo that can be understood, tuned, and run by a small engineering team without additional backend dependencies.

---

## 2. Top-Level Dataflow

At a high level, the pipeline is:

`public sources -> raw_input -> ingest -> automated analysis -> score aggregation -> review/label -> manifest/index/export`

Concretely:

1. Public source videos are fetched and clipped into a flat `raw_input/` directory.
2. A sidecar `raw_input/source_map.csv` carries provenance and `demo_category`.
3. `ingest.py` probes each clip with `ffprobe` and merges source metadata.
4. `sampler.py` extracts evenly spaced keyframes and sampled frames.
5. `manual_precheck.py` computes quality heuristics.
6. `stability_qa.py` computes camera motion / shake severity.
7. `wide_lens_qa.py` estimates whether the framing is consistent with egocentric wide-angle capture.
8. `hands_qa.py` uses MediaPipe Hand Landmarker to produce hand and egocentric features.
9. `gate.py` and `curation.py` aggregate all of the above into public scores and statuses.
10. `reviewer.py` and `labeler.py` provide human override and metadata enrichment.
11. `report.py` exports manifest, JSON, thumbnails, SQLite, summary, and curated clips.

---

## 3. Input Acquisition and Provenance Model

The repo intentionally uses a flat `raw_input/` directory rather than category subfolders. That constraint is handled by a provenance sidecar:

- `raw_input/*.mp4`
- `raw_input/source_map.csv`

Each row in `source_map.csv` includes:

- `filename`
- `source_title`
- `source_url`
- `license`
- `attribution`
- `demo_category`
- `clip_start_sec`
- `clip_duration_sec`

### Why this matters

This is a useful product choice because it decouples:

- physical file layout
- source attribution
- benchmark/demo category labeling

That is cleaner than overloading directory structure or filenames with provenance semantics.

### Demo source generation

`scripts/fetch_public_demo_inputs.py` is the acquisition layer. It:

1. pulls public source files from Wikimedia Commons
2. caches the originals under `.cache/source_videos/`
3. cuts shorter demo clips with `ffmpeg`
4. writes `raw_input/source_map.csv`

This makes the demo self-contained without checking full original source archives into the repo.

---

## 4. Ingest Layer: `ingest.py`

The ingest layer is a pure metadata + validation stage.

### Technique: `ffprobe`

Each file is probed with:

- `-show_streams`
- `-show_format`
- JSON output

The probe output is used to extract:

- raw width / height
- display width / height after rotation normalization
- duration
- frame count if available
- effective FPS
- codec
- file size

### Rotation-aware dimension normalization

If the stream reports a `rotation` side-data value or rotate tag of `90` or `270`, the pipeline swaps width and height to compute display dimensions. This is important because raw container metadata can understate the true spatial resolution of portrait or rotated captures.

### Effective frame rate

The pipeline prefers:

`effective_fps = nb_frames / duration`

If frame count is unavailable, it falls back to:

`r_frame_rate`

This is a practical choice. For curation purposes, effective delivered frame rate is more useful than container-declared nominal rate.

### Hashing

Each clip gets an MD5 hash for lightweight identity and repeatability checks.

### Input validation

Two baseline format checks are applied at ingest:

- minimum resolution
- minimum frame rate

These are still implemented through the original internal `c1/c2` booleans, but the public-facing surface has been reframed away from criterion naming.

---

## 5. Frame Sampling Strategy: `sampler.py`

The pipeline uses two complementary frame-sampling patterns.

### A. Evenly spaced sparse seeks

Used for:

- keyframe extraction
- hand analysis
- wide-lens analysis

Mechanism:

- compute `N` evenly spaced timestamps over clip duration
- seek to each timestamp
- decode a frame

This gives broad temporal coverage while bounding compute.

### B. Sequential sampled decode

Used for:

- manual precheck
- stability scans

Mechanism:

- open video with OpenCV
- compute a decode step from source FPS and target sample FPS
- advance with `grab()`
- materialize only sampled frames with `read()`

This is materially faster than repeated random access on longer clips and avoids high memory usage.

### Key design insight

The sampling subsystem is intentionally not one-size-fits-all. Sparse seek is better for global coverage when a fixed count of representative frames is enough. Sequential decode is better when temporal continuity matters.

---

## 6. Manual Quality Precheck: `manual_precheck.py`

This module is not the final judge. It is an assistive quality-analysis pass that produces evidence for downstream scoring and human review.

### Metrics computed

#### Brightness

For each sampled frame:

- convert to grayscale
- compute mean luma

Flag:

- frame is dark if mean luma is below `MANUAL_DARK_LUMA_THRESHOLD`

Derived outputs:

- `avg_luma`
- `dark_ratio`
- `dark_segments`

#### Blur

For each sampled frame:

- compute Laplacian variance on grayscale

Low Laplacian variance implies weak high-frequency detail and therefore blur.

Derived outputs:

- `avg_blur_laplacian_var`
- `blurry_ratio`
- `blurry_segments`

#### Jump-cut / scene-cut heuristic

For each frame:

- build a normalized grayscale histogram
- compare adjacent frame histograms with Bhattacharyya distance

If distance exceeds `MANUAL_SCENE_CUT_THRESHOLD`, it becomes a jump-cut candidate.

Derived outputs:

- `max_cut_score`
- `candidate_jump_cuts`

#### Static / duplicate heuristic

This is one of the newer public-demo additions.

The module computes:

1. low-resolution grayscale thumbnails
2. per-frame mean absolute intensity difference
3. a binary perceptual-like hash from thresholding the thumbnail around its mean
4. Hamming distance ratio between consecutive hashes

From that it derives:

- `motion_score`
- `static_ratio`
- `duplicate_ratio`
- `static_duplicate_score`

This is a lightweight but effective way to identify:

- near-static holds
- repeated imagery
- clips with low temporal novelty

It is not robust enough to replace full duplicate search at scale, but it is entirely appropriate for a local curation demo.

---

## 7. Hand and Egocentric Feature Stack: `hands_qa.py`

This is the most egocentric-specific subsystem in the repo.

### Backend

The required backend is MediaPipe Hand Landmarker.

The implementation:

- auto-downloads the `.task` model if absent
- uses the MediaPipe Tasks API
- runs in video mode when possible
- processes RGB frames after OpenCV decode

### Preprocessing

Before inference:

- frames are resized to a configured width
- optional CLAHE support exists for lighting robustness

### Detection loop

For each sampled frame, the model produces:

- hand count
- handedness/confidence
- landmark sets per hand

The code then derives:

- `mp_detected`
- `mp_confidence`
- per-hand centroids
- frame-level centroid
- region buckets

### Skin-tone fallback

If the policy is “at least one hand” and MediaPipe fails to detect a hand, the pipeline can apply a broad HSV skin-mask fallback as a recovery path for low-confidence misses.

This is a pragmatic recall-oriented fallback, not a replacement for the model.

### Temporal smoothing

The raw hand-present booleans are passed through a median filter. This fills isolated false negatives without introducing heavy state modeling.

### Missing-hand segmentation

The pipeline tracks contiguous windows where the final hand-present signal is false and exports them as missing-hand segments with start/end timestamps. These segments are used both for scoring context and for manual inspection.

### Public hand-derived features

The demo extends the original report into a more explicit feature surface:

#### `hand_visible_ratio`

This is the fraction of sampled frames in which hands are considered present after smoothing and fallback.

#### `hand_presence_score`

This is the mean detection confidence across sampled frames.

#### `hand_motion_score`

This is derived from average centroid displacement across consecutive detected frames, normalized by `HAND_MOTION_NORMALIZER`.

Interpretation:

- low score: hands are static or mostly absent
- high score: hands move actively through the scene

#### `hand_count_distribution`

A distribution over frames containing:

- `0`
- `1`
- `2_plus`

This gives useful signal about interaction density and framing style.

#### `hand_region_distribution`

The frame centroid is bucketed spatially:

- horizontal: `left`, `center`, `right`
- vertical: `upper`, `middle`, `lower`

This becomes a compact egocentric framing descriptor.

#### `egocentric_proxy_score`

This is a hand-crafted composite using:

- hand visibility ratio
- lower-frame occupancy
- center-frame occupancy

The intuition is simple: egocentric manipulation video often places hands in the lower and central portions of the frame.

### Why this is strong for a demo

This subsystem delivers a lot of egocentric signal without training a custom model:

- presence
- persistence
- motion
- count
- approximate spatial priors

That is a good tradeoff for a candidate evaluation or lightweight production prototype.

---

## 8. Perspective / Wide-Lens Analysis: `wide_lens_qa.py`

This subsystem estimates whether the clip’s optics and framing are consistent with egocentric wide-angle capture.

This is not semantic camera-view classification. It is geometric heuristic analysis.

### Core signals

#### A. Line curvature from Hough lines

Pipeline:

1. grayscale conversion
2. blur + Canny edges
3. probabilistic Hough line extraction
4. sample offsets along line normals
5. fit and remove linear trend
6. measure residual bend energy

This estimates whether nominally straight scene lines bow in a way consistent with barrel distortion.

#### B. Edge-density falloff

The frame is split into:

- center region
- periphery ring

The ratio of edge density between center and periphery is used as a proxy for wide-angle / vignetting-like characteristics.

#### C. Radial distortion score

The system:

- extracts edge points in an outer ring
- computes image gradients
- compares local gradient orientation to the radial direction from frame center

Tangential edge orientation at the periphery is treated as evidence of barrel distortion.

### Aggregation

The module combines:

- radial distortion signal
- edge falloff
- line curvature

into `wide_lens_score` plus confidence.

### Decision logic

The decision layer is nuanced:

- pass if score and confidence exceed configured thresholds
- salvage some low-score cases if edge/line evidence is still strong
- hard-fail low-confidence close-up / non-wide-lens cases
- return borderline when the signal is uncertain

This is a useful design because it avoids forcing every ambiguous case into pass/fail. The pipeline preserves uncertainty and routes it into review when appropriate.

---

## 9. Stability Analysis: `stability_qa.py`

This module is the most computationally sophisticated piece of the repo.

It uses a two-stage motion analysis design to balance speed and fidelity.

### Stage 1: cheap probe

The probe scans distributed windows over the full clip using:

- phase correlation between grayscale frames
- low-resolution analysis
- coarse jitter scoring

This gives a cheap approximation of where strong motion spikes may exist.

### Window construction

Probe windows are:

- distributed over clip duration
- jittered deterministically per clip
- merged if overlapping

This ensures broad temporal coverage without scanning every frame of every second.

### Stage 2: targeted refine

Only the highest-risk windows go through heavier analysis based on:

- Shi-Tomasi feature selection
- Lucas-Kanade optical flow
- affine partial transform estimation

From these, the system estimates:

- translation magnitude
- rotation-equivalent displacement
- robust motion severity

### Motion severity model

The motion score blends:

- median displacement
- a capped motion model from estimated transform

This is a strong engineering choice because it reduces sensitivity to outliers while still capturing large global camera motion.

### Stability outputs

The module exports:

- `max_motion_score`
- `peak_motion_score`
- `spike_segments`
- total shaky seconds
- coverage metadata
- whether the analysis was truncated or requires manual confirmation

### Decision logic

Hard fail:

- sufficiently long super-shaky segments
- severe early spikes

Borderline:

- partial scans requiring confirmation

Pass:

- stable clips
- brief shaky bursts below failure duration thresholds

This subsystem is designed well for curation because it distinguishes:

- genuinely unusable shake
- acceptable bursty motion
- uncertainty due to limited analysis coverage

---

## 10. Aggregation and Public Scoring: `gate.py` + `curation.py`

The gate layer still orchestrates the original automated flow, but the public demo wraps it in a cleaner curation schema.

### Internal automated decision flow

Order of operations:

1. ingest + format checks
2. keyframe extraction
3. manual precheck
4. stability analysis
5. perspective analysis
6. hand analysis when needed
7. aggregate row assembly

### Early exits

The pipeline intentionally short-circuits expensive work when:

- format checks already fail
- strong perspective or stability hard-fail evidence already exists
- borderline stability should be routed to manual review instead of spending more compute on hands

This is throughput-aware design, not just correctness-driven design.

### Public scoring model

The public `training_value_score` is a weighted heuristic composite defined in YAML.

Current weights:

- `hand_visible_ratio`: `0.20`
- `hand_presence_score`: `0.12`
- `hand_motion_score`: `0.12`
- `egocentric_proxy_score`: `0.10`
- `brightness_score`: `0.10`
- `blur_score`: `0.10`
- `motion_score`: `0.08`
- `camera_stability_score`: `0.10`
- `static_duplicate_score`: `0.08`

The score is weight-normalized after aggregation.

### Derived quality scores

The aggregation layer translates raw evidence into bounded scores:

- `brightness_score = 1 - dark_ratio`
- `blur_score = 1 - blurry_ratio`
- `motion_score` from precheck inter-frame differences
- `camera_stability_score` from inverse-normalized max motion
- `static_duplicate_score` from inverse-normalized duplication/staticity

### Public flags and reasons

The system distinguishes:

#### `quality_flags`

Compact machine-readable conditions such as:

- `low_resolution`
- `low_frame_rate`
- `low_hand_visibility`
- `third_person_likely`
- `unstable_camera`
- `blur`
- `low_brightness`
- `static_or_duplicate`

#### `curation_reasons`

Higher-level rationale strings capturing why a clip landed where it did, including propagated automated details and review-state reasons like `manual_review_pending`.

### Public status mapping

Current public status thresholds:

- `recommended >= 0.68`
- `low_value <= 0.38`
- otherwise `needs_review`

But the status layer also respects hard overrides:

- hard automated failures map to `low_value`
- manual pending routes auto-pass clips to `needs_review`
- manual approval upgrades to `recommended`
- manual rejection forces `low_value`

That is the correct product behavior. The automated score informs triage, but human decisions remain authoritative.

---

## 11. Manual Review Layer: `reviewer.py`

This subsystem is explicitly designed as a curation UI for ambiguous or uncertain cases.

### Inputs available to the reviewer

For each clip, the operator gets:

- basic metadata
- contact sheet
- review summary
- hand metrics
- wide-lens / perspective signals
- stability signals
- manual precheck quality evidence

### Review logic

The reviewer resolves:

- content quality
- egocentric fit
- borderline automated signals

The review record is stored as:

- `manual_review.json`

with:

- `approved: true/false`
- a list of reject / low-value reasons

### Why this matters

The pipeline does not pretend that all ambiguity can be solved automatically. It provides a structured operator override system without building a separate application stack.

---

## 12. Labeling Layer: `labeler.py`

For recommended clips, the labeling surface captures:

- `task`
- `objects`
- `task_outcome`
- `environment`
- `time_of_day`
- `notes`

This is intentionally human-entered. The repo does not currently implement automated action recognition or object detection, and preserving that boundary is the right technical choice for a small demo.

This means the system differentiates between:

- automated curation signals
- operator-provided semantic task labels

That separation is clean and product-correct.

---

## 13. Export and Index Layer: `report.py`

The export layer converts internal artifacts into public deliverables.

### Outputs

`exports/` contains:

- `dataset_manifest.csv`
- `dataset_manifest.json`
- `dataset_index.sqlite`
- `summary.json`
- `thumbnails/*.jpg`
- `clips/*.mp4` for recommended clips

### Manifest contents

Each row includes:

- clip identity and provenance
- public status
- `training_value_score`
- `quality_flags`
- `curation_reasons`
- hand-derived features
- quality and stability metrics
- optional manual review state
- optional semantic labels

### Thumbnails

The first available extracted keyframe is copied into `exports/thumbnails/`.

### Curated clip export

Only `recommended` clips are physically copied into `exports/clips/`.

This is a subtle but good product decision: the export set is already filtered down to the clips most suitable for downstream training use.

### SQLite index

The manifest is mirrored into SQLite with indexes on:

- `status`
- `demo_category`
- `training_value_score`
- `quality_flags`
- `curation_reasons`

That turns the export into something queryable immediately by scripts, notebooks, or simple review tooling.

---

## 14. Configuration Layer: `config.py` + `config/default.yaml`

Configuration is split into:

- Python defaults in `config.py`
- public overrides in `config/default.yaml`

This has two important effects:

1. The code remains runnable if YAML loading fails or a config file is absent.
2. Public thresholds and weights are externally visible and adjustable.

The config surface includes:

- ingest thresholds
- hand-tracking settings
- perspective thresholds
- stability thresholds
- manual precheck thresholds
- public status thresholds
- public score weights
- export paths
- demo categories

This is the right balance between hardcoded determinism and demo tunability.

---

## 15. Runtime and Dependency Model

The runtime is intentionally lightweight:

- Python 3.11
- OpenCV
- MediaPipe
- NumPy / SciPy
- pandas
- PyYAML
- Click
- questionary
- ffmpeg / ffprobe

There is no required:

- database server
- message queue
- remote inference service
- GPU
- cloud storage backend

That makes the repo easy to run in interviews, internal demos, or lightweight prototyping contexts.

---

## 16. What Is Technically Strong Here

From a CTO perspective, the strongest aspects are:

### A. Bounded complexity

The repo solves a meaningful curation problem without introducing unnecessary infrastructure.

### B. Good signal composition

It combines:

- geometry
- motion
- perceptual sharpness
- lighting
- hand activity
- review state

into a coherent scoring surface.

### C. Auditable heuristics

Every outcome can be traced to explicit features and thresholds.

### D. Clear operator handoff

The system knows where automation ends and where human review should take over.

### E. Practical packaging

The public export surface is immediately usable:

- manifest
- JSON
- SQLite
- thumbnails
- curated clips

This is what makes it feel like a real product demo rather than a collection of isolated scripts.

---

## 17. Current Limitations

The current system is strong as a deterministic demo, but there are obvious ceilings:

### 1. Perspective heuristic fragility

The wide-lens / perspective logic is geometry-based and can misclassify legitimate egocentric footage if optical cues are weak or atypical.

### 2. No semantic video understanding

The system does not yet infer:

- action class
- object interaction type
- task success automatically

Those remain manual.

### 3. Static/duplicate detection is local, not corpus-global

The current heuristic is frame-adjacent within a clip, not cross-dataset deduplication.

### 4. Threshold tuning is heuristic

The YAML config is transparent, but not statistically calibrated on a large benchmark.

### 5. Review is still CLI-driven

This is acceptable for a demo, but a production review experience would likely move into a thin web UI.

---

## 18. If This Were Extended Toward Production

The highest-leverage next steps would be:

1. Replace or augment perspective heuristics with a learned egocentric / third-person classifier
2. Add corpus-level duplicate search using embeddings or perceptual hashing
3. Learn score calibration from labeled curator decisions
4. Add automatic task / object interaction priors
5. Add a thin review UI on top of the SQLite + artifact structure

The current architecture is suitable for these upgrades because the pipeline is already modular and artifact-driven.

---

## 19. Bottom Line

This repo demonstrates good engineering judgment:

- it preserves working code
- introduces public configurability
- adds meaningful egocentric features on top of MediaPipe
- avoids overbuilding
- cleanly separates automated scoring from human review
- exports artifacts in formats that are immediately useful

If the goal is to evaluate whether someone can take an existing codebase, preserve what works, reposition it into a different product shape, and ship an end-to-end technical demo with concrete outputs, this repo is a credible example of that skill set.
