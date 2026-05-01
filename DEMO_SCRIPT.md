# One-Minute Technical Demo Script

This demo takes raw public video and turns it into a scored, indexed, reviewed, and exported training dataset.

The first stage is ingest. We probe each clip with `ffprobe`, normalize resolution, frame-rate, duration, and provenance metadata, and persist that into per-clip artifacts.

The second stage is automated curation. MediaPipe Hand Landmarker stays in place as the hand backend, and we derive `hand_visible_ratio`, `hand_presence_score`, `hand_motion_score`, hand-count distributions, and a simple egocentric proxy from the same detections. In parallel, the existing quality stack stays active: blur, brightness, motion, camera stability, and a lightweight static-or-duplicate heuristic built from sampled-frame similarity.

The third stage is aggregation. Those signals roll into a configurable `training_value_score`, plus public-facing `quality_flags` and `curation_reasons`. We keep thresholds and weights in YAML so the demo is tunable on public or self-recorded data without hardcoded private rules.

The last stage is review and export. Borderline clips can still go through the existing manual review and labeling flow, and the final output is a dataset manifest in CSV and JSON, thumbnails, recommended clip exports, a summary, and a SQLite index for filtering by status, score, flags, reasons, and category.
