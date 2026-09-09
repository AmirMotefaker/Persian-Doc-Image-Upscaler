from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_PIXELS = 40_000_000


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"فرمت پشتیبانی نمی‌شود: {path.suffix or 'بدون پسوند'}")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("ابعاد تصویر بیش از حد مجاز است.")
        image = image.convert("RGB")
        rgb = np.asarray(image)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_png(path: str | Path, image: np.ndarray) -> str:
    path = str(path)
    if not cv2.imwrite(path, image):
        raise RuntimeError("ذخیره خروجی تصویر ناموفق بود.")
    return path
