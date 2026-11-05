import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from body import estimate_measurements
from classify import classifier

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
WARDROBE_DIR = ROOT / "wardrobe"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

for d in (WARDROBE_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Virtual Fitroom")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MeasureRequest(BaseModel):
    world_landmarks: list[list[float]] = Field(..., min_length=33, max_length=33)
    landmarks: Optional[list[list[float]]] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/api/wardrobe")
def wardrobe():
    items = []
    for category in ("upper", "lower", "dresses"):
        cat_dir = WARDROBE_DIR / category
        if not cat_dir.is_dir():
            continue
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if img_path.stem.endswith("_cut"):
                continue
            meta_path = img_path.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
            cutout_path = img_path.with_name(f"{img_path.stem}_cut.png")
            cutout_url = (
                f"/wardrobe/{category}/{cutout_path.name}" if cutout_path.exists() else None
            )
            items.append(
                {
                    "id": f"{category}/{img_path.stem}",
                    "name": meta.get("name", img_path.stem),
                    "category": meta.get("category", category),
                    "image_url": f"/wardrobe/{category}/{img_path.name}",
                    "cutout_url": cutout_url,
                    "size_chart": meta.get("size_chart"),
                }
            )
    return {"items": items}


@app.post("/api/measure")
def measure(req: MeasureRequest):
    try:
        result = estimate_measurements(req.world_landmarks, req.landmarks)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.post("/api/classify")
async def classify(image: UploadFile = File(...)):
    from PIL import Image
    import io

    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read image")
    try:
        result = classifier.predict(img)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result


@app.post("/api/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment_id: str = Form(...),
    steps: int = Form(20),
    height: int = Form(768),
    width: int = Form(576),
):
    from io import BytesIO

    from PIL import Image

    from masking import cloth_mask

    garment_img = resolve_garment_image(garment_id)
    if garment_img is None:
        raise HTTPException(status_code=404, detail=f"Unknown garment: {garment_id}")
    category = garment_id.split("/")[0]

    try:
        person_pil = Image.open(BytesIO(await person.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read person image")

    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA GPU required for AI try-on")

    try:
        with TRYON_LOCK:
            from catvton import get_pipeline

            pipe = get_pipeline()
            mask = cloth_mask(person_pil, category)
            result = pipe(
                person_pil,
                garment_img,
                mask,
                num_inference_steps=max(8, min(50, steps)),
                height=height,
                width=width,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU out of memory — try fewer steps")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Try-on failed: {e}")

    out_name = f"tryon_{time.strftime('%Y%m%d_%H%M%S')}.png"
    result.save(RESULTS_DIR / out_name)
    return {"result_url": f"/results/{out_name}", "category": category}


def resolve_garment_image(garment_id: str):
    from PIL import Image

    parts = garment_id.split("/")
    if len(parts) != 2:
        return None
    category, stem = parts
    cat_dir = WARDROBE_DIR / category
    if not cat_dir.is_dir():
        return None
    cutout = cat_dir / f"{stem}_cut.png"
    if cutout.exists():
        im = Image.open(cutout).convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = cat_dir / f"{stem}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


TRYON_LOCK = threading.Lock()


app.mount("/wardrobe", StaticFiles(directory=WARDROBE_DIR), name="wardrobe")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/models", StaticFiles(directory=FRONTEND_DIR / "models"), name="mlmodels")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
