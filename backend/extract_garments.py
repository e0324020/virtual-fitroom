"""Extract garment-only cutouts from on-model wardrobe photos.

Uses a SegFormer model fine-tuned on ATR clothing parsing
(mattmdjaga/segformer_b2_clothes) to segment the worn garment, then writes a
transparent PNG containing only the garment pixels (person + background removed).

The live overlay then warps a real garment cutout instead of a photo of a model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ROOT = Path(__file__).resolve().parent.parent
WARDROBE_DIR = ROOT / "wardrobe"

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# ATR label ids: 4=Upper-clothes, 5=Skirt, 6=Pants, 7=Dress
CATEGORY_LABELS = {
    "upper": [4],
    "lower": [5, 6],
    "dresses": [7],
}

# If the primary label is missing (e.g. a "dress" the model parsed as two pieces),
# fall back to the union of plausible garment labels.
FALLBACK_LABELS = {
    "upper": [4, 7],
    "lower": [5, 6, 4],
    "dresses": [4, 5, 6, 7],
}


def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, processor, device


def segment(im, model, processor, device):
    inputs = processor(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=im.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(1)[0].cpu().numpy()


def clean_mask(mask):
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n > 1:
        kept = np.zeros_like(mask, dtype=np.uint8)
        min_area = 0.02 * mask.size
        for i in range(1, n):
            comp = labels == i
            if comp.sum() >= min_area:
                kept[comp] = 255
        mask = kept

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def cutout(img_path, out_path, model, processor, device, labels):
    im = Image.open(img_path).convert("RGB")
    seg = segment(im, model, processor, device)
    mask = np.isin(seg, labels).astype(np.uint8) * 255
    mask = clean_mask(mask)

    ys, xs = np.where(mask > 32)
    if len(xs) == 0:
        return False

    pad = 12
    x0, x1 = max(0, xs.min() - pad), min(im.width - 1, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(im.height - 1, ys.max() + pad)

    rgba = np.dstack([np.array(im), mask])
    crop = rgba[y0 : y1 + 1, x0 : x1 + 1]
    Image.fromarray(crop, "RGBA").save(out_path)
    return True


def main():
    model, processor, device = build_model()
    print(f"Segmentation model ready on {device}")

    total = 0
    for category, labels in CATEGORY_LABELS.items():
        cat_dir = WARDROBE_DIR / category
        if not cat_dir.is_dir():
            continue
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if img_path.stem.endswith("_cut"):
                continue
            out_path = img_path.with_name(f"{img_path.stem}_cut.png")
            try:
                ok = cutout(img_path, out_path, model, processor, device, labels)
                if not ok:
                    print(f"  {img_path.name}: primary labels empty, retrying fallback…")
                    ok = cutout(
                        img_path, out_path, model, processor, device, FALLBACK_LABELS[category]
                    )
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {img_path.name}: {e}")
                continue
            status = "ok" if ok else "no garment found"
            print(f"  {img_path.name} -> {out_path.name} ({status})")
            total += int(ok)

    print(f"Done. {total} cutouts written.")


if __name__ == "__main__":
    main()
