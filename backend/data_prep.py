"""Download a garment dataset subset for classifier training + wardrobe samples.

Source: PestoRosso/lamoda-fashion-product-images (high-res parquet shards on HF)
Classes: upper (Topwear), lower (Bottomwear), dress (Dresses)

Images are extracted from parquet shards (~285 MB each) and resized, so only a
handful of shards is needed for a balanced 3-class subset.
"""

import argparse
import io
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "garments"
SHARD_CACHE = ROOT / "data" / ".hf_cache"
WARDROBE_DIR = ROOT / "wardrobe"

REPO_ID = "PestoRosso/lamoda-fashion-product-images"
N_SHARDS = 45

CLASSES = {"upper": "Topwear", "lower": "Bottomwear", "dress": "Dress"}
WARDROBE_FOLDER = {"upper": "upper", "lower": "lower", "dress": "dresses"}


def disk_counts():
    return {k: len(list((DATA_DIR / k).glob("*.jpg"))) for k in CLASSES}


def extract_image_bytes(value):
    if isinstance(value, dict):
        return value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def process_shard(shard_index, sub_to_key, counts, target, max_per_class):
    fname = f"data/train_part_{shard_index:04d}.parquet"
    print(f"downloading shard {shard_index}...", flush=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=fname,
        repo_type="dataset",
        cache_dir=str(SHARD_CACHE),
    )
    added = {k: 0 for k in CLASSES}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=128):
        cols = batch.to_pydict()
        sub_cats = cols.get("sub_category") or []
        ids = cols.get("id") or []
        images = cols.get("image") or []
        for sub, item_id, img_val in zip(sub_cats, ids, images):
            key = sub_to_key.get(sub)
            if not key or counts[key] >= target:
                continue
            raw = extract_image_bytes(img_val)
            if not raw:
                continue
            dest = DATA_DIR / key / f"{key}_{item_id}.jpg"
            if dest.exists():
                counts[key] += 1
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img.thumbnail((720, 960), Image.LANCZOS)
                img.save(dest, quality=90)
                counts[key] += 1
                added[key] += 1
            except Exception as e:
                print(f"    skip {item_id}: {e}")
        if all(counts[k] >= target for k in CLASSES):
            break
    print(f"shard {shard_index}: added {added}, totals {counts}", flush=True)
    return counts


def collect(target_per_class):
    for k in CLASSES:
        (DATA_DIR / k).mkdir(parents=True, exist_ok=True)
    counts = disk_counts()
    print(f"starting counts: {counts}", flush=True)
    if all(counts[k] >= target_per_class for k in CLASSES):
        print("already enough images", flush=True)
        return counts

    sub_to_key = {v: k for k, v in CLASSES.items()}
    for shard in range(N_SHARDS):
        if all(counts[k] >= target_per_class for k in CLASSES):
            break
        counts = process_shard(shard, sub_to_key, counts, target_per_class, None)
    print(f"final counts: {counts}", flush=True)
    return counts


def copy_wardrobe_samples(n_per_class):
    for class_key in CLASSES:
        src_dir = DATA_DIR / class_key
        dst_dir = WARDROBE_DIR / WARDROBE_FOLDER[class_key]
        dst_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(src_dir.glob("*.jpg"))[:n_per_class]
        for i, img_path in enumerate(images):
            dst = dst_dir / f"{class_key}_{i + 1}.jpg"
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((640, 850), Image.LANCZOS)
            img.save(dst, quality=92)
        print(f"[wardrobe] {class_key}: {len(images)} samples -> {dst_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--wardrobe-per-class", type=int, default=8)
    parser.add_argument("--wardrobe-only", action="store_true")
    args = parser.parse_args()

    if not args.wardrobe_only:
        collect(args.per_class)
    copy_wardrobe_samples(args.wardrobe_per_class)
    print("done", flush=True)
