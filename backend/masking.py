"""Cloth-agnostic mask generation for CatVTON try-on.

Reuses the SegFormer clothing-parsing model to build the region that should be
replaced by the selected garment category.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# ATR label ids
UPPER_CLOTHES, SKIRT, PANTS, DRESS = 4, 5, 6, 7

CATEGORY_LABELS = {
    "upper": [UPPER_CLOTHES],
    "lower": [SKIRT, PANTS],
    "dresses": [UPPER_CLOTHES, SKIRT, PANTS, DRESS],
}

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device).eval()
    return _model, _processor


def cloth_mask(person: Image.Image, category: str, dilate: int = 15) -> Image.Image:
    """Return a binary (0/255) 'L' mask covering the clothing region to replace."""
    model, processor = _load()
    device = next(model.parameters()).device

    im = person.convert("RGB")
    inputs = processor(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=im.size[::-1], mode="bilinear", align_corners=False
    )
    seg = up.argmax(1)[0].cpu().numpy()

    labels = CATEGORY_LABELS.get(category, [UPPER_CLOTHES])
    mask = np.isin(seg, labels).astype(np.uint8) * 255

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, k, iterations=1)

    # fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) > 500:
            cv2.drawContours(filled, [c], -1, 255, thickness=-1)
    mask = np.maximum(mask, filled)

    return Image.fromarray(mask, mode="L")
