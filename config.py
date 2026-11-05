"""Configuration for the Virtual Fitting Room application."""

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS_TARGET = 30

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.5

# Skeletal landmark indices (MediaPipe Pose)
LANDMARK_SHOULDER_LEFT = 11
LANDMARK_SHOULDER_RIGHT = 12
LANDMARK_CHEST = 11  # Mid-shoulder approximation
LANDMARK_WAIST_LEFT = 23
LANDMARK_WAIST_RIGHT = 24
LANDMARK_EAR_LEFT = 7
LANDMARK_EAR_RIGHT = 8

GARMENT_KEY = {
    "shoulder_left": LANDMARK_SHOULDER_LEFT,
    "shoulder_right": LANDMARK_SHOULDER_RIGHT,
    "waist_left": LANDMARK_WAIST_LEFT,
    "waist_right": LANDMARK_WAIST_RIGHT,
}

SIZE_THRESHOLDS = {
    "S": {"shoulder_max": 350, "waist_max": 350},
    "M": {"shoulder_max": 450, "waist_max": 450},
    "L": {"shoulder_max": float("inf"), "waist_max": float("inf")},
}

# Distance-invariant sizing: shoulder-to-head ratio (distance between ears).
# These ratios are broadly stable regardless of camera distance.
RATIO_THRESHOLDS = {
    "S": 2.3,
    "M": 2.9,
    "L": float("inf"),
}

GARMENT_OFFSET = 10  # Offset from shoulder / waist landmarks (pixels)

# Fitting parameters
GARMENT_WIDTH_FACTOR = 1.1    # garment shoulder width = shoulder span * this (aspect preserved)
GARMENT_MAX_WIDTH_FACTOR = 1.6  # hard cap: garment width <= shoulder span * this
GARMENT_NECK_OFFSET = 0.06    # garment top sits this fraction of the torso above the shoulders
GARMENT_HEIGHT_FACTOR = 1.2   # legacy (unused)
GARMENT_BOTTOM_FACTOR = 1.45  # legacy (unused)
NECK_OFFSET = 0.04            # legacy (unused)
MASK_THRESHOLD = 0.40         # person-segmentation probability threshold

COLORS = {
    "line": (0, 255, 0),
    "point": (0, 0, 255),
    "text": (255, 255, 255),
    "text_bg": (0, 0, 0),
    "overlay": (128, 128, 128),
}