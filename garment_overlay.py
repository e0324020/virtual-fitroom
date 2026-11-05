"""Garment overlay engine using dynamic geometric mapping.

Two rendering modes:
  - perspective (default): fit-mapping. Preserves the garment's aspect
    ratio, scales it to the detected shoulder span, and rotates it to match
    the shoulder line angle so it drapes naturally (handles tilted bodies
    without stretching the fabric).
  - rectangle: legacy mode using cvzone.overlayPNG (simple scale + position).
"""

import math

import cv2
import cvzone
import numpy as np

import config


class GarmentOverlay:
    """Maps a garment PNG onto detected body landmarks."""

    def __init__(self, garment_path=None, mode="perspective"):
        """Initialize overlay.

        Args:
            garment_path: Path to a transparent PNG garment (optional here).
            mode: "perspective" or "rectangle".
        """
        self.mode = mode
        self.img_garment = None
        self._dy = 0.0      # fine tune: vertical shift (px)
        self._dx = 0.0      # fine tune: horizontal shift (px)
        self._scale = 1.0   # fine tune: extra scale multiplier
        if garment_path:
            self.load(garment_path)

    def reset_fit(self):
        """Clear manual fit adjustments."""
        self._dy = 0.0
        self._dx = 0.0
        self._scale = 1.0

    @property
    def fit(self):
        """Current manual fit adjustments (dy, dx, scale)."""
        return self._dy, self._dx, self._scale

    def load(self, garment_path):
        """(Re)load a garment asset as a BGRA image."""
        self.img_garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        if self.img_garment is None:
            raise ValueError(f"Could not load garment image: {garment_path}")

    def overlay_garment(self, img, lm_list, body_mask=None):
        """Overlay the garment onto the torso.

        Args:
            img: BGR frame.
            lm_list: Full 33-landmark pixel list.
            body_mask: Optional person-segmentation mask (float 0-1, HxW)
                used to clip the garment to the body silhouette so it
                appears "worn" rather than floating over the background.

        Returns:
            Frame with the garment drawn.
        """
        if len(lm_list) != 33 or self.img_garment is None:
            return img

        if self.mode == "perspective":
            return self._perspective_overlay(img, lm_list, body_mask)
        return self._rectangle_overlay(img, lm_list, body_mask)

    # ------------------------------------------------------------------ #
    #  Perspective mode (fit-mapping - preserves garment proportions)     #
    # ------------------------------------------------------------------ #
    def _first_opaque_row(self):
        """Index of the topmost row that carries visible garment pixels."""
        alpha = self.img_garment[:, :, 3]
        rows = np.where((alpha > 40).sum(axis=1) > 0)[0]
        return int(rows.min()) if len(rows) else 0

    def _reference_width(self):
        """Width of the garment at its shoulder/bust area.

        Scaling by the full canvas width undersizes garments that have
        sleeves or transparent padding, because only part of that width is
        actually the body. We measure the widest opaque row in the top 40%
        of the garment instead and match THAT to the user's shoulders.
        """
        alpha = self.img_garment[:, :, 3]
        mask = alpha > 40
        ys = np.where(mask.any(axis=1))[0]
        if len(ys) == 0:
            return self.img_garment.shape[1]
        top, bot = int(ys.min()), int(ys.max())
        limit = top + max(int((bot - top) * 0.40), 1)
        widths = mask[top:limit].sum(axis=1)
        ref = int(widths.max()) if len(widths) else 0
        return ref if ref > 0 else self.img_garment.shape[1]

    @staticmethod
    def _shade_wrap(resized):
        """Darken the left/right edges to fake the fabric wrapping the torso."""
        h, w = resized.shape[:2]
        xs = np.linspace(-1.0, 1.0, w)
        profile = 1.0 - 0.30 * (xs ** 2)          # 1.0 center -> 0.70 edges
        resized[:, :, :3] = np.clip(
            resized[:, :, :3].astype(np.float32) * profile[None, :, None],
            0, 255,
        ).astype(resized.dtype)
        return resized

    def _perspective_overlay(self, img, lm_list, body_mask=None):
        shoulder_l = np.array(lm_list[config.LANDMARK_SHOULDER_LEFT], dtype=np.float32)
        shoulder_r = np.array(lm_list[config.LANDMARK_SHOULDER_RIGHT], dtype=np.float32)
        hip_l = np.array(lm_list[config.LANDMARK_WAIST_LEFT], dtype=np.float32)
        hip_r = np.array(lm_list[config.LANDMARK_WAIST_RIGHT], dtype=np.float32)

        g_h, g_w = self.img_garment.shape[:2]
        if g_w < 1 or g_h < 1:
            return img

        shoulder_w = float(np.linalg.norm(shoulder_r - shoulder_l))
        shoulder_mid = (shoulder_l + shoulder_r) / 2.0
        hip_mid = (hip_l + hip_r) / 2.0
        # Vertical shoulder->hip distance (torso length).
        torso_len = abs(float(hip_mid[1] - shoulder_mid[1]))
        if shoulder_w < 5 or torso_len < 5:
            return img

        # ---- Size: match the garment's shoulder/bust width to the user's --
        # shoulders, preserving aspect ratio so dresses stay long.
        ref_w = self._reference_width()
        scale = shoulder_w * config.GARMENT_WIDTH_FACTOR / ref_w * self._scale
        # Never let a garment (e.g. a strappy dress) balloon past the shoulders.
        scale = min(scale, shoulder_w * config.GARMENT_MAX_WIDTH_FACTOR / g_w)
        new_w = max(int(round(g_w * scale)), 1)
        new_h = max(int(round(g_h * scale)), 1)
        resized = cv2.resize(self.img_garment, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        resized = self._shade_wrap(resized)

        # ---- Rotate to match the shoulder tilt -----------------------------
        angle = math.degrees(math.atan2(
            shoulder_r[1] - shoulder_l[1], shoulder_r[0] - shoulder_l[0]))
        # Wrap spurious near-180 angles (left/right landmark swaps) so the
        # garment never renders upside down.
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        cx, cy = new_w // 2, new_h // 2
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        rot_w = int(new_h * abs(matrix[0, 1]) + new_w * abs(matrix[0, 0]))
        rot_h = int(new_h * abs(matrix[0, 0]) + new_w * abs(matrix[0, 1]))
        matrix[0, 2] += rot_w / 2.0 - cx
        matrix[1, 2] += rot_h / 2.0 - cy

        rotated = cv2.warpAffine(
            resized, matrix, (rot_w, rot_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # ---- Placement: hang the garment's top row from the shoulder line --
        src_top = self._first_opaque_row()
        anchor = matrix @ np.array([new_w / 2.0, src_top * scale, 1.0])
        top_y = shoulder_mid[1] - config.GARMENT_NECK_OFFSET * torso_len
        x1 = int(round(shoulder_mid[0] - rot_w / 2.0 + self._dx))
        y1 = int(round(top_y - anchor[1] + self._dy))

        return self._blend(img, rotated, x1, y1, body_mask)

    @staticmethod
    def _blend(frame, overlay, x1, y1, body_mask=None):
        """Alpha-composite ``overlay`` (BGRA) onto ``frame`` at (x1, y1).

        When ``body_mask`` (frame-sized float 0-1) is given, the garment
        alpha is multiplied by the person's silhouette so the clothing only
        covers the body and never spills into the background.
        """
        fh, fw = frame.shape[:2]
        oh, ow = overlay.shape[:2]

        x0_src = max(0, x1)
        y0_src = max(0, y1)
        x1_src = min(fw, x1 + ow)
        y1_src = min(fh, y1 + oh)
        if x1_src <= x0_src or y1_src <= y0_src:
            return frame

        ox0 = x0_src - x1
        oy0 = y0_src - y1
        ox1 = ox0 + (x1_src - x0_src)
        oy1 = oy0 + (y1_src - y0_src)

        region = overlay[oy0:oy1, ox0:ox1]
        alpha = region[:, :, 3].astype(np.float32) / 255.0

        if body_mask is not None:
            mask = body_mask[y0_src:y1_src, x0_src:x1_src].astype(np.float32)
            # Feather the silhouette edge so the garment merges into the body
            # instead of showing a hard, floating cut-out.
            mask = cv2.GaussianBlur(mask, (0, 0), 4.0)
            alpha = alpha * mask

        alpha = alpha[..., None]
        color = region[:, :, :3].astype(np.float32)
        under = frame[y0_src:y1_src, x0_src:x1_src].astype(np.float32)

        # ---- Relight the garment with the body's own brightness -----------
        # Multiplying by the underlying luminance makes the fabric pick up
        # the real scene's shadows and highlights, so it looks worn instead
        # of being a flat sticker pasted on top.
        lum = cv2.cvtColor(under.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = float(lum.mean())
        if mean > 5.0:
            light = np.clip(lum / mean, 0.65, 1.35)[..., None]
            color = color * light

        frame[y0_src:y1_src, x0_src:x1_src] = (
            color * alpha + under * (1.0 - alpha)
        ).astype(np.uint8)
        return frame

    # ------------------------------------------------------------------ #
    #  Rectangle mode (legacy cvzone approach)                            #
    # ------------------------------------------------------------------ #
    def _rectangle_overlay(self, img, lm_list, body_mask=None):
        shoulder_l = np.array(lm_list[config.LANDMARK_SHOULDER_LEFT], dtype=np.float32)
        shoulder_r = np.array(lm_list[config.LANDMARK_SHOULDER_RIGHT], dtype=np.float32)
        hip_l = np.array(lm_list[config.LANDMARK_WAIST_LEFT], dtype=np.float32)
        hip_r = np.array(lm_list[config.LANDMARK_WAIST_RIGHT], dtype=np.float32)

        shoulder_w = float(np.linalg.norm(shoulder_r - shoulder_l))
        shoulder_mid = (shoulder_l + shoulder_r) / 2.0
        hip_mid = (hip_l + hip_r) / 2.0
        torso_len = abs(float(hip_mid[1] - shoulder_mid[1]))
        if shoulder_w < 1 or torso_len < 1:
            return img

        g_h, g_w = self.img_garment.shape[:2]
        ref_w = self._reference_width()
        scale = shoulder_w * config.GARMENT_WIDTH_FACTOR / ref_w * self._scale
        scale = min(scale, shoulder_w * config.GARMENT_MAX_WIDTH_FACTOR / g_w)
        new_w = max(int(round(g_w * scale)), 1)
        new_h = max(int(round(g_h * scale)), 1)
        garment_scaled = cv2.resize(self.img_garment, (new_w, new_h),
                                    interpolation=cv2.INTER_LINEAR)
        garment_scaled = self._shade_wrap(garment_scaled)

        src_top = self._first_opaque_row()
        top_y = shoulder_mid[1] - config.GARMENT_NECK_OFFSET * torso_len
        x1 = int(round(shoulder_mid[0] - new_w / 2.0 + self._dx))
        y1 = int(round(top_y - src_top * scale + self._dy))

        return self._blend(img, garment_scaled, x1, y1, body_mask)