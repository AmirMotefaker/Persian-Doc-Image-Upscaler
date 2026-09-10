from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/"
    "master/models/FSRCNN_x2.pb"
)
MODEL_NAME = "FSRCNN_x2.pb"


def _model_path() -> Path:
    root = Path(
        os.environ.get(
            "DAQIQKHAN_MODEL_DIR",
            Path.home() / ".cache" / "daqiqkhan" / "models",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / MODEL_NAME


def _ensure_model() -> Path:
    path = _model_path()
    if path.is_file() and path.stat().st_size > 10_000:
        return path

    tmp = path.with_suffix(".download")
    print(f"[SR] downloading {MODEL_NAME}...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if not tmp.is_file() or tmp.stat().st_size <= 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("فایل مدل Super-Resolution معتبر دانلود نشد.")
    tmp.replace(path)
    print(f"[SR] model ready: {path}", flush=True)
    return path


@lru_cache(maxsize=1)
def _engine():
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "ماژول OpenCV dnn_superres نصب نیست؛ requirements جدید را نصب کنید."
        )
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(_ensure_model()))
    sr.setModel("fsrcnn", 2)
    return sr


def super_resolve_visual(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Visual-only SR path. OCR never depends on these generated pixels."""
    if image is None or image.size == 0:
        raise ValueError("تصویر ورودی Super-Resolution معتبر نیست.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ساختار تصویر برای Super-Resolution پشتیبانی نمی‌شود.")

    upscaled = _engine().upsample(image)

    # Keep the SR result crisp without strong halos around Persian dots/stems.
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    local = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8)).apply(lightness)
    lab[:, :, 0] = cv2.addWeighted(lightness, 0.45, local, 0.55, 0)
    upscaled = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(upscaled, (0, 0), 0.7)
    upscaled = cv2.addWeighted(upscaled, 1.14, blur, -0.14, 0)

    target_scale = max(1.0, float(scale))
    if abs(target_scale - 2.0) > 0.01:
        height, width = image.shape[:2]
        target = (
            max(1, round(width * target_scale)),
            max(1, round(height * target_scale)),
        )
        interpolation = (
            cv2.INTER_LANCZOS4
            if target_scale > 2.0
            else cv2.INTER_AREA
        )
        upscaled = cv2.resize(upscaled, target, interpolation=interpolation)

    return upscaled
