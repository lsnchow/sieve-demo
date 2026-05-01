"""Central configuration: thresholds, paths, and public demo scoring settings."""

from __future__ import annotations

from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional until runtime install
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")
EXPORT_DIR = REPO_ROOT / "exports"
RAW_INPUT_DIR = REPO_ROOT / "raw_input"
DEMO_SOURCE_MAP_NAME = "source_map.csv"

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ogv"}

# ---------------------------------------------------------------------------
# C1: Resolution
# ---------------------------------------------------------------------------
MIN_WIDTH = 1280
MIN_HEIGHT = 720

# ---------------------------------------------------------------------------
# C2: Frame rate
# ---------------------------------------------------------------------------
MIN_FPS = 29.0

# ---------------------------------------------------------------------------
# C3a: Hands visible
# ---------------------------------------------------------------------------
HANDS_SAMPLE_FPS = 1.0
HANDS_MIN_SAMPLE_FPS = 0.15
HANDS_TARGET_FRAMES = 300
HANDS_DYNAMIC_SAMPLE_FPS = True
HANDS_RESAMPLE_FPS = 1.5
HANDS_MIN_RATIO = 0.91
HANDS_BORDERLINE_LOW = 0.90
HANDS_BORDERLINE_HIGH = 0.98
HANDS_LONG_CLIP_NO_RESAMPLE_SEC = 120
HANDS_REQUIRED_COUNT = 1

MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_PRESENCE_CONFIDENCE = 0.5
MEDIAPIPE_MAX_NUM_HANDS = 2
HAND_LANDMARKER_MODEL = REPO_ROOT / "models" / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HANDS_SMOOTHING_WINDOW = 5
HANDS_PREPROCESS_WIDTH = 640
HANDS_ENABLE_CLAHE = False
HANDS_USE_VIDEO_MODE = True
HAND_MOTION_NORMALIZER = 0.08

SKIN_HSV_LOWER = (0, 20, 70)
SKIN_HSV_UPPER = (35, 255, 255)
SKIN_MIN_AREA_RATIO = 0.005

# ---------------------------------------------------------------------------
# C3b: Wide-lens
# ---------------------------------------------------------------------------
WIDE_LENS_SAMPLE_FRAMES = 10
WIDE_LENS_MAX_DIM = 480
WIDE_LENS_CANNY_LOW = 50
WIDE_LENS_CANNY_HIGH = 150
WIDE_LENS_HOUGH_THRESHOLD = 60
WIDE_LENS_LINE_MIN_LENGTH_RATIO = 0.12
WIDE_LENS_LINE_MAX_GAP = 10
WIDE_LENS_LINE_SAMPLE_STEP_PX = 6
WIDE_LENS_LINE_SCAN_NORMAL_PX = 4
WIDE_LENS_MIN_LINES_FOR_CONF = 8
WIDE_LENS_WEIGHT_RADIAL = 0.45
WIDE_LENS_WEIGHT_EDGE = 0.20
WIDE_LENS_WEIGHT_LINE = 0.35
WIDE_LENS_PASS_THRESHOLD = 0.35
WIDE_LENS_BORDERLINE_LOW = 0.35
WIDE_LENS_CONFIDENCE_MIN = 0.5
WIDE_LENS_CLOSEUP_SCORE_MAX = 0.40
WIDE_LENS_CLOSEUP_CONF_MAX = 0.78
WIDE_LENS_LOW_CONF_FAIL_SCORE_MAX = 0.45
WIDE_LENS_LOW_CONF_FAIL_CONF_MAX = 0.40
WIDE_LENS_SALVAGE_SCORE_MIN = 0.31
WIDE_LENS_SALVAGE_CONFIDENCE_MIN = 0.80
WIDE_LENS_SALVAGE_EDGE_FALLOFF_MIN = 1.0
WIDE_LENS_SALVAGE_LINE_CONF_MIN = 0.30

# ---------------------------------------------------------------------------
# C9: Stability
# ---------------------------------------------------------------------------
STABILITY_SAMPLE_FPS = 2
STABILITY_MIN_SAMPLE_FPS = 1.0
STABILITY_TARGET_FRAME_PAIRS = 900
STABILITY_MAX_ANALYSIS_SEC = 3600.0
STABILITY_WINDOW_SEC = 1.0
STABILITY_SPIKE_THRESHOLD = 130.0
STABILITY_BORDERLINE_LOW = 20.0
STABILITY_MIN_FAIL_SEGMENT_SEC = 1.0
STABILITY_MAX_FEATURES = 500
STABILITY_FEATURE_QUALITY = 0.01
STABILITY_EARLY_HARD_FAIL = True
STABILITY_TRUNCATED_FAIL_TOTAL_SHAKY_SEC = 4.0
STABILITY_EARLY_SPIKE_WINDOW_SEC = 30.0
STABILITY_EARLY_SPIKE_SCORE_FAIL = 150.0

STABILITY_COARSE_ENABLED = True
STABILITY_COARSE_MAX_DIM = 320
STABILITY_COARSE_SAMPLE_FPS = 1.0
STABILITY_COARSE_TARGET_FRAMES = 120
STABILITY_COARSE_TRIGGER = 16.0
STABILITY_COARSE_PERCENTILE = 85.0
STABILITY_MAX_CANDIDATE_WINDOWS = 6
STABILITY_CANDIDATE_WINDOW_SEC = 1.5
STABILITY_CANDIDATE_PAD_SEC = 0.25
STABILITY_CONTROL_WINDOW_COUNT = 1

STABILITY_PROBE_COVERAGE = 0.08
STABILITY_PROBE_WINDOW_SEC = 1.5
STABILITY_PROBE_K_MIN = 4
STABILITY_PROBE_K_MAX = 40
STABILITY_PROBE_FPS = 8.0
STABILITY_PROBE_MAX_DIM = 320
STABILITY_PROBE_JITTER_FRACTION = 0.15

STABILITY_REFINE_TOP_FRACTION = 0.25
STABILITY_REFINE_MIN_WINDOWS = 2
STABILITY_REFINE_MAX_WINDOWS = 8
STABILITY_REFINE_COVERAGE_FRACTION = 0.03
STABILITY_REFINE_FPS = 2.0
STABILITY_REFINE_MAX_DIM = 640

# ---------------------------------------------------------------------------
# Manual quality helpers
# ---------------------------------------------------------------------------
MANUAL_PRECHECK_SAMPLE_FPS = 2.0
MANUAL_DARK_LUMA_THRESHOLD = 45.0
MANUAL_BLUR_LAPLACIAN_THRESHOLD = 80.0
MANUAL_SCENE_CUT_THRESHOLD = 0.55
MANUAL_MAX_CUT_CANDIDATES = 30
MANUAL_STATIC_DIFF_THRESHOLD = 0.02
MANUAL_DUPLICATE_HAMMING_RATIO_MAX = 0.06
MANUAL_DARK_RATIO_FLAG = 0.45
MANUAL_BLURRY_RATIO_FLAG = 0.45
STATIC_DUPLICATE_FLAG_SCORE = 0.60

# ---------------------------------------------------------------------------
# Caching / execution
# ---------------------------------------------------------------------------
USE_CACHED_REPORTS = True
GATE_EARLY_EXIT_ON_HARD_FAIL = True

# ---------------------------------------------------------------------------
# Runtime profiling
# ---------------------------------------------------------------------------
ENABLE_STAGE_TIMING = True
WRITE_TIMING_REPORT = True

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
WRITE_RATIONALE_REPORTS = False

# ---------------------------------------------------------------------------
# Gate classification
# ---------------------------------------------------------------------------
GATE_PASS = "PASS_AUTOGATE"
GATE_FAIL = "FAIL_AUTOGATE"
GATE_BORDERLINE = "BORDERLINE_REVIEW"

# ---------------------------------------------------------------------------
# Public demo curation
# ---------------------------------------------------------------------------
PUBLIC_STATUS_RECOMMENDED = "recommended"
PUBLIC_STATUS_NEEDS_REVIEW = "needs_review"
PUBLIC_STATUS_LOW_VALUE = "low_value"

STATUS_THRESHOLDS = {
    "recommended": 0.68,
    "low_value": 0.38,
}

TRAINING_VALUE_WEIGHTS = {
    "hand_visible_ratio": 0.20,
    "hand_presence_score": 0.12,
    "hand_motion_score": 0.12,
    "egocentric_proxy_score": 0.10,
    "brightness_score": 0.10,
    "blur_score": 0.10,
    "motion_score": 0.08,
    "camera_stability_score": 0.10,
    "static_duplicate_score": 0.08,
}

CAMERA_STABILITY_NORMALIZER = 140.0

CLIP_EXPORT_STATUSES = [PUBLIC_STATUS_RECOMMENDED]

# ---------------------------------------------------------------------------
# Labeling controlled vocabulary
# ---------------------------------------------------------------------------
ENVIRONMENTS = [
    "kitchen", "living_room", "bedroom", "bathroom", "office",
    "garage", "yard", "garden", "workshop", "store",
    "restaurant", "street", "park", "gym", "classroom",
    "warehouse", "laundry_room", "hallway", "balcony", "other",
]

TASK_OUTCOMES = ["success", "fail", "partial"]
TIMES_OF_DAY = ["daytime", "nighttime", "indoor"]

DEMO_CATEGORIES = [
    "high_value_pov",
    "pov_no_action",
    "third_person",
    "low_quality",
    "static_or_duplicate",
    "ambiguous",
]

# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------
KEYFRAME_COUNT = 16
CONTACT_SHEET_COLS = 4

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

_SERIALIZABLE_KEYS = [
    "ARTIFACTS_DIR",
    "EXPORT_DIR",
    "RAW_INPUT_DIR",
    "DEMO_SOURCE_MAP_NAME",
    "VIDEO_EXTENSIONS",
    "MIN_WIDTH",
    "MIN_HEIGHT",
    "MIN_FPS",
    "HANDS_SAMPLE_FPS",
    "HANDS_MIN_SAMPLE_FPS",
    "HANDS_TARGET_FRAMES",
    "HANDS_RESAMPLE_FPS",
    "HANDS_MIN_RATIO",
    "HANDS_BORDERLINE_LOW",
    "HANDS_BORDERLINE_HIGH",
    "HANDS_REQUIRED_COUNT",
    "HAND_LANDMARKER_MODEL",
    "HAND_LANDMARKER_MODEL_URL",
    "HAND_MOTION_NORMALIZER",
    "WIDE_LENS_PASS_THRESHOLD",
    "WIDE_LENS_BORDERLINE_LOW",
    "STABILITY_SPIKE_THRESHOLD",
    "STABILITY_MIN_FAIL_SEGMENT_SEC",
    "MANUAL_DARK_RATIO_FLAG",
    "MANUAL_BLURRY_RATIO_FLAG",
    "STATIC_DUPLICATE_FLAG_SCORE",
    "STATUS_THRESHOLDS",
    "TRAINING_VALUE_WEIGHTS",
    "PUBLIC_STATUS_RECOMMENDED",
    "PUBLIC_STATUS_NEEDS_REVIEW",
    "PUBLIC_STATUS_LOW_VALUE",
    "CLIP_EXPORT_STATUSES",
    "DEMO_CATEGORIES",
]


def _coerce_like(current, new_value):
    if isinstance(current, Path):
        return Path(new_value)
    if isinstance(current, set):
        return set(new_value)
    if isinstance(current, tuple):
        return tuple(new_value)
    return new_value


def load_yaml_config(config_path: Path | None = None) -> bool:
    """Load optional YAML overrides into module globals."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if yaml is None or not path.exists():
        return False

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")

    namespace = globals()
    for key, value in data.items():
        if key not in namespace:
            continue
        namespace[key] = _coerce_like(namespace[key], value)
    return True


def get_public_config() -> dict:
    """Return a JSON-serializable config payload for CLI display."""
    payload = {}
    namespace = globals()
    for key in _SERIALIZABLE_KEYS:
        value = namespace[key]
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, set):
            payload[key] = sorted(value)
        else:
            payload[key] = value
    return payload
