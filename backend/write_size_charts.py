"""Write size-chart JSON sidecars for wardrobe garments.

Standard women's apparel chart (cm). Each <image>.jpg gets a matching
<image>.json with a display name, category and size chart.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WARDROBE_DIR = ROOT / "wardrobe"

SIZE_CHART = [
    {"size": "XS", "chest_min": 74, "chest_max": 80, "waist_min": 58, "waist_max": 64, "hip_min": 82, "hip_max": 88},
    {"size": "S", "chest_min": 80, "chest_max": 86, "waist_min": 64, "waist_max": 70, "hip_min": 88, "hip_max": 94},
    {"size": "M", "chest_min": 86, "chest_max": 92, "waist_min": 70, "waist_max": 76, "hip_min": 94, "hip_max": 100},
    {"size": "L", "chest_min": 92, "chest_max": 98, "waist_min": 76, "waist_max": 82, "hip_min": 100, "hip_max": 106},
    {"size": "XL", "chest_min": 98, "chest_max": 104, "waist_min": 82, "waist_max": 88, "hip_min": 106, "hip_max": 112},
]

CATEGORY_LABEL = {"upper": "Top", "lower": "Bottom", "dresses": "Dress"}
FOLDER_TO_CATEGORY = {"upper": "upper", "lower": "lower", "dresses": "dress"}


def main():
    for folder, category in FOLDER_TO_CATEGORY.items():
        cat_dir = WARDROBE_DIR / folder
        if not cat_dir.is_dir():
            continue
        for img_path in sorted(cat_dir.glob("*.jpg")):
            meta = {
                "name": f"{CATEGORY_LABEL[folder]} {img_path.stem.split('_')[-1]}",
                "category": category,
                "size_chart": SIZE_CHART,
            }
            meta_path = img_path.with_suffix(".json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"wrote {meta_path.name}")


if __name__ == "__main__":
    main()
