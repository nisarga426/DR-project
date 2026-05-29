import cv2
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def crop_retinal_circle(img: np.ndarray) -> np.ndarray:
    """Detect the retinal circle and crop tightly around it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def apply_ben_graham(img: np.ndarray, sigma: int = 30) -> np.ndarray:
    """Ben Graham illumination normalization."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    normalized = cv2.addWeighted(img, 4, blurred, -4, 128)
    return normalized

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the green channel (highest contrast for DR)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def blur_score(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Below 50 = blurry."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def preprocess_image(src_path: str, dst_path: str, size: int = 512) -> float:
    """Full pipeline: load → crop → CLAHE → Ben Graham → resize → save."""
    img = cv2.imread(src_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {src_path}")
    img = crop_retinal_circle(img)
    img = apply_clahe(img)
    img = apply_ben_graham(img)
    img = cv2.resize(img, (size, size))
    score = blur_score(img)
    cv2.imwrite(dst_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return score

def run_aptos(raw_dir: str, out_dir: str, size: int = 512) -> None:
    """Process all APTOS 2019 training images."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_dir / "train.csv")
    records = []
    for i, row in df.iterrows():
        src = str(raw_dir / "train_images" / (row["id_code"] + ".png"))
        dst = str(out_dir / (row["id_code"] + ".png"))
        try:
            score = preprocess_image(src, dst, size)
            records.append({"image_id": row["id_code"],
                            "label": row["diagnosis"],
                            "quality": round(score, 2),
                            "keep": score > 50})
        except Exception as e:
            log.warning(f"Skipping {row['id_code']}: {e}")
        if i % 200 == 0:
            log.info(f"Processed {i}/{len(df)} images...")
    meta = pd.DataFrame(records)
    meta.to_csv(out_dir / "processed_labels.csv", index=False)
    kept = meta["keep"].sum()
    log.info(f"Done! {kept}/{len(meta)} images passed quality filter.")

if __name__ == "__main__":
    run_aptos(
        raw_dir="data/raw/aptos2019",
        out_dir="data/processed/train",
    )
