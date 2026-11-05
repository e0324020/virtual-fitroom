"""Manages a collection of garments for multi-garment switching.

Scans the garments directory for apparel images, lazily loads overlays,
and tracks the currently selected garment.
"""

import os

from garment_overlay import GarmentOverlay

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")


class GarmentManager:
    """Handles the garment library, selection and switching."""

    def __init__(self, garments_dir, mode="perspective"):
        self.garments_dir = garments_dir
        self.mode = mode
        self.names = []
        self.paths = []
        self._overlays = {}
        self.current = 0
        self._scan()

    def _scan(self):
        """Populate the garment list from the garments directory."""
        if not os.path.isdir(self.garments_dir):
            return
        for filename in sorted(os.listdir(self.garments_dir)):
            if filename.lower().endswith(SUPPORTED_EXTS):
                self.names.append(filename)
                self.paths.append(os.path.join(self.garments_dir, filename))

    def _load(self, index):
        """Load (and cache) the overlay for the garment at ``index``."""
        overlay = self._overlays.get(index)
        if overlay is None:
            overlay = GarmentOverlay(self.paths[index], self.mode)
            self._overlays[index] = overlay
        return overlay

    # ------------------------------------------------------------------ #
    @property
    def count(self):
        return len(self.names)

    @property
    def current_name(self):
        if not self.names:
            return "None"
        return self.names[self.current]

    def get_active(self):
        """Return the overlay for the currently selected garment."""
        if not self.paths:
            return None
        if self.current >= len(self.paths):
            self.current = 0
        return self._load(self.current)

    def switch(self, index):
        """Select garment ``index`` (wraps around). Returns True on success."""
        if not self.paths:
            return False
        self.current = index % len(self.paths)
        return True

    def cycle(self, step=1):
        """Move the selection by ``step`` garments (wraps around)."""
        if not self.paths:
            return False
        self.current = (self.current + step) % len(self.paths)
        return True

    def overlay(self, img, lm_list, body_mask=None):
        """Apply the selected garment overlay to ``img``."""
        active = self.get_active()
        if active is None:
            return img
        return active.overlay_garment(img, lm_list, body_mask)

    def adjust(self, dy=0.0, dx=0.0, scale_factor=1.0):
        """Apply a manual fit adjustment to the active overlay."""
        active = self.get_active()
        if active is None:
            return
        active._dy += dy
        active._dx += dx
        active._scale = max(0.3, min(active._scale * scale_factor, 3.0))

    def reset_active_fit(self):
        """Reset the manual fit of the active overlay."""
        active = self.get_active()
        if active is not None:
            active.reset_fit()

    def active_fit_text(self):
        """Human-readable fit adjustment summary for the HUD."""
        active = self.get_active()
        if active is None:
            return ""
        dy, dx, sc = active.fit
        return f"adj dy={dy:+.0f} dx={dx:+.0f} scale={sc:.2f}"