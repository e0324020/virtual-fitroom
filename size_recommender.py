"""Automated size recommendation engine.

Maps pixel distances between skeletal landmarks (shoulder span, waist span)
to standard retail sizes (S, M, L). Pixel measurements alone depend on the
camera distance, so body spans are normalized against head width (the
ear-to-ear distance), producing a distance-invariant ratio that correlates
with physical frame size:

  shoulder_ratio = shoulder_span / ear_span

An average adult has shoulder_span roughly 2.3-2.9x their head width;
larger frames push this ratio above ~2.9.
"""

import config


class SizeRecommender:
    """Computes recommended garment size from landmark distances."""

    def __init__(self, thresholds=None):
        self._pixel_thresholds = thresholds or config.SIZE_THRESHOLDS
        self._ratio_thresholds = config.RATIO_THRESHOLDS

    def recommend(self, lm_list):
        """Return the recommended size given the 33-landmark list.

        Args:
            lm_list: Full 33-landmark pixel list.

        Returns:
            Tuple (size_label, metrics_dict).
        """
        if len(lm_list) != 33:
            return "N/A", {}

        shoulder_l = lm_list[config.LANDMARK_SHOULDER_LEFT]
        shoulder_r = lm_list[config.LANDMARK_SHOULDER_RIGHT]
        waist_l = lm_list[config.LANDMARK_WAIST_LEFT]
        waist_r = lm_list[config.LANDMARK_WAIST_RIGHT]
        ear_l = lm_list[config.LANDMARK_EAR_LEFT]
        ear_r = lm_list[config.LANDMARK_EAR_RIGHT]

        def dist(a, b):
            return abs(a[0] - b[0])

        shoulder_px = dist(shoulder_r, shoulder_l)
        waist_px = dist(waist_r, waist_l)
        torso_px = abs(waist_l[1] - shoulder_l[1])
        head_px = max(dist(ear_r, ear_l), 1.0)

        # Distance-invariant ratio classification
        shoulder_ratio = shoulder_px / head_px
        size = "S"
        for label in ("S", "M", "L"):
            if shoulder_ratio <= self._ratio_thresholds[label]:
                size = label
                break

        # Pixel-only reference (assumes fixed camera framing) as fallback
        pixel_ref = self._pixel_reference(shoulder_px, waist_px)

        # Favour the ratio-based estimate, but fall back to the pixel
        # reference if the head landmarks were unreliable (zero-width head).
        if head_px <= 1:
            size = pixel_ref

        metrics = {
            "shoulder_px": shoulder_px,
            "waist_px": waist_px,
            "torso_px": torso_px,
            "head_px": head_px,
            "shoulder_ratio": round(shoulder_ratio, 3),
            "pixel_size_ref": pixel_ref,
        }
        return size, metrics

    def _pixel_reference(self, shoulder_px, waist_px):
        """Pixel-only size reference (assumes fixed camera distance)."""
        if shoulder_px <= self._pixel_thresholds["S"]["shoulder_max"] and \
                waist_px <= self._pixel_thresholds["S"]["waist_max"]:
            return "S"
        if shoulder_px <= self._pixel_thresholds["M"]["shoulder_max"] and \
                waist_px <= self._pixel_thresholds["M"]["waist_max"]:
            return "M"
        return "L"

    def recommend_closest(self, lm_list, available_sizes):
        """Return the closest available size if it exists.

        Args:
            lm_list: Full 33-landmark pixel list.
            available_sizes: Iterable of size labels on offer.

        Returns:
            Best-matching available size label.
        """
        size, _ = self.recommend(lm_list)
        if size in available_sizes:
            return size

        order = ["S", "M", "L"]
        if size not in order:
            return "N/A"

        # Walk outward from the recommended size to find an offer
        idx = order.index(size)
        candidates = list(available_sizes)
        candidates.sort(key=lambda s: abs(order.index(s) - idx))
        return candidates[0] if candidates else "N/A"