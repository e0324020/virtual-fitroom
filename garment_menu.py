"""On-screen dropdown menu for selecting garments with the mouse."""

import cv2

import config


class GarmentMenu:
    """A clickable dropdown listing available garments.

    Draws a panel over the frame; left-clicking an item selects it.
    Supports keyboard cycling independently via the calling code.
    """

    def __init__(self, names, x=10, y=70, width=240, item_h=32):
        self.names = names
        self.x, self.y = x, y
        self.width = width
        self.item_h = item_h
        self.max_items = 8
        self.selected = 0
        self.scroll = 0
        self.visible = True

    # ------------------------------------------------------------------ #
    #  Rendering                                                          #
    # ------------------------------------------------------------------ #
    def draw(self, img):
        """Draw the dropdown panel on ``img``. Returns the frame."""
        if not self.visible or not self.names:
            return img

        count = min(len(self.names), self.max_items)
        panel_h = self.item_h * (count + 1) + 6

        h, w = img.shape[:2]
        right = self.x + self.width + 16
        bottom = self.y + panel_h + 6
        if right < w and bottom < h:
            x0, y0, x1, y1 = self.x, self.y, right, bottom
        else:
            x0 = max(w - self.width - 16, 0)
            y0 = max(h - panel_h - 6, 0)
            x1 = min(x0 + self.width + 16, w)
            y1 = min(y0 + panel_h + 6, h)

        # Panel background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), config.COLORS["overlay"], cv2.FILLED)
        img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

        # Header
        header = "Garments  [G] hide"
        cv2.putText(
            img, header, (x0 + 8, y0 + self.item_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS["text"], 1, cv2.LINE_AA,
        )
        cv2.line(img, (x0, y0 + self.item_h), (x1, y0 + self.item_h),
                 config.COLORS["text"], 1)

        # Items
        visible = self.names[self.scroll:self.scroll + self.max_items]
        for i, name in enumerate(visible):
            item_top = y0 + self.item_h + i * self.item_h
            actual_idx = self.scroll + i

            if actual_idx == self.selected:
                cv2.rectangle(
                    img, (x0, item_top), (x1, item_top + self.item_h),
                    config.COLORS["point"], cv2.FILLED,
                )

            number = f"{actual_idx + 1}."
            cv2.putText(
                img, number, (x0 + 8, item_top + self.item_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLORS["text"], 1, cv2.LINE_AA,
            )
            label = name if len(name) <= 22 else name[:19] + "..."
            cv2.putText(
                img, label, (x0 + 34, item_top + self.item_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLORS["text"], 1, cv2.LINE_AA,
            )
        return img

    # ------------------------------------------------------------------ #
    #  Interaction                                                        #
    # ------------------------------------------------------------------ #
    def hit_test(self, px, py):
        """Return True if (px, py) is inside the visible panel."""
        if not self.visible or not self.names:
            return False
        x0, y0 = self.x, self.y
        panel_h = self.item_h * (min(len(self.names), self.max_items) + 1) + 12
        x1 = x0 + self.width + 16
        y1 = y0 + panel_h
        return x0 <= px < x1 and y0 <= py < y1

    def select_at(self, px, py):
        """Select the garment at the clicked position. Returns True if hit."""
        if not self.hit_test(px, py):
            return False
        item_y = self.y + self.item_h
        rel = py - item_y
        if rel < 0:
            return True  # clicked the header - ignore
        idx = self.scroll + rel // self.item_h
        if 0 <= idx < len(self.names):
            self.selected = idx
            return True
        return False

    def select_by_key(self, number_key):
        """Select the garment matching a 1-9 key. Returns True if valid."""
        idx = number_key - 1
        if 0 <= idx < len(self.names):
            self.selected = idx
            self.scroll = max(0, idx - self.max_items + 1)
            return True
        return False

    def scroll_up(self):
        if self.scroll > 0:
            self.scroll -= 1

    def scroll_down(self):
        if self.scroll < len(self.names) - self.max_items:
            self.scroll += 1