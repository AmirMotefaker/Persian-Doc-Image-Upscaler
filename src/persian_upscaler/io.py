from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
OUTPUT_FORMATS = {"PNG": ".png", "JPG": ".jpg", "WEBP": ".webp"}
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


def save_image(
    directory: str | Path,
    stem: str,
    image: np.ndarray,
    output_format: str = "PNG",
) -> str:
    fmt = output_format.upper()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"فرمت خروجی پشتیبانی نمی‌شود: {output_format}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}{OUTPUT_FORMATS[fmt]}"

    params: list[int] = []
    if fmt == "JPG":
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif fmt == "WEBP":
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]

    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError("ذخیره خروجی تصویر ناموفق بود.")
    return str(path)


def save_png(path: str | Path, image: np.ndarray) -> str:
    """Compatibility wrapper for internal OCR previews."""
    path = Path(path)
    return save_image(path.parent, path.stem, image, "PNG")
