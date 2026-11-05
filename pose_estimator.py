"""Pose estimation module using MediaPipe's PoseLandmarker (Tasks API).

MediaPipe 1.x replaced the legacy ``mp.solutions.pose`` API with the Tasks
framework. A CNN-based pose landmarker processes each frame and returns the
33 skeletal landmarks used by the fitting room.
"""

import time

import cv2
import mediapipe as mp
import numpy as np

import config

# Skeleton edges (MediaPipe Pose topology)
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (27, 29), (29, 31), (27, 28), (26, 28), (28, 30), (30, 32),
    (25, 26),
)

_VISION = mp.tasks.vision
_BASE = mp.tasks.BaseOptions
_RUNNING_MODE = _VISION.RunningMode


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker for 33-landmark tracking."""

    def __init__(self, model_path="models/pose_landmarker_lite.task"):
        options = _VISION.PoseLandmarkerOptions(
            base_options=_BASE(model_asset_path=model_path),
            running_mode=_RUNNING_MODE.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE,
            output_segmentation_masks=True,
        )
        self._landmarker = _VISION.PoseLandmarker.create_from_options(options)
        self._landmarker_options = options

        self._timestamp_ms = 0
        self._inference_time = 0.0
        self._screen_t0 = time.perf_counter()
        # Normalized (x, y) landmarks when a person is detected, else None.
        self._landmarks_norm = None
        # Person segmentation probability mask (float 0-1) or None.
        self._seg_mask = None

    @property
    def fps(self):
        """Frames per second based on the last inference time."""
        if self._inference_time <= 0:
            return 0.0
        return 1.0 / self._inference_time

    @property
    def detected(self):
        return self._landmarks_norm is not None

    @property
    def seg_mask(self):
        """Person segmentation mask resized to frame size, or None."""
        return self._seg_mask

    def find_pose(self, img, draw=True):
        """Detect pose landmarks in a BGR image.

        Args:
            img: BGR frame.
            draw: Whether to draw the skeleton on the frame.

        Returns:
            img: Frame with the skeleton drawn (if draw=True).
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Video mode requires a monotonically increasing timestamp (ms).
        self._timestamp_ms = int((time.perf_counter() - self._screen_t0) * 1000)

        start = cv2.getTickCount()
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._inference_time = (cv2.getTickCount() - start) / cv2.getTickFrequency()

        if result.pose_landmarks:
            self._landmarks_norm = [
                (landmark.x, landmark.y) for landmark in result.pose_landmarks[0]
            ]
        else:
            self._landmarks_norm = None

        # Person silhouette mask (same resolution as the input frame).
        masks = getattr(result, "segmentation_masks", None)
        if masks:
            mask = masks[0]
            if mask is not None:
                if hasattr(mask, "numpy_view"):
                    mask = mask.numpy_view()
                mask = np.asarray(mask, dtype=np.float32)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = cv2.resize(
                    mask, (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Sharpen the probability map around the threshold so the
                # garment clips cleanly to the body silhouette.
                t = config.MASK_THRESHOLD
                mask = np.clip((mask - t) / max(1.0 - t, 1e-6), 0.0, 1.0)
            self._seg_mask = mask
        else:
            self._seg_mask = None

        if draw and self._landmarks_norm is not None:
            self._draw_skeleton(img)

        return img

    def get_position(self, img, landmark_index=None, draw=True):
        """Return pixel landmark coordinates.

        Args:
            img: Frame to measure against.
            landmark_index: Optional; if given, only returns/highlights one.
            draw: Draw the requested point as a circle.

        Returns:
            Full 33-entry [x, y] list (may be empty when no person detected).
        """
        h, w, _ = img.shape
        lm_list = []
        if self._landmarks_norm is None:
            return lm_list

        lm_list = [
            [round(x * w), round(y * h)] for x, y in self._landmarks_norm
        ]
        if landmark_index is not None and draw:
            x, y = lm_list[landmark_index]
            cv2.circle(img, (x, y), 5, config.COLORS["point"], cv2.FILLED)
        return lm_list

    def _draw_skeleton(self, img):
        """Draw the pose skeleton using normalized landmark coordinates."""
        h, w, _ = img.shape

        def to_px(landmark):
            return round(landmark[0] * w), round(landmark[1] * h)

        for conn in POSE_CONNECTIONS:
            cv2.line(
                img,
                to_px(self._landmarks_norm[conn[0]]),
                to_px(self._landmarks_norm[conn[1]]),
                config.COLORS["line"],
                2,
                cv2.LINE_AA,
            )
        for landmark in self._landmarks_norm:
            cv2.circle(img, to_px(landmark), 3, config.COLORS["point"], cv2.FILLED)

    def close(self):
        """Release MediaPipe resources."""
        self._landmarker.close()