from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

EDSR_URL = (
    "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/"
    "master/models/EDSR_x4.pb"
)
EDSR_NAME = "EDSR_x4.pb"
FSRCNN_URL = (
    "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/"
    "master/models/FSRCNN_x2.pb"
)
FSRCNN_NAME = "FSRCNN_x2.pb"


def _model_root() -> Path:
    root = Path(
        os.environ.get(
            "DAQIQKHAN_MODEL_DIR",
            Path.home() / ".cache" / "daqiqkhan" / "models",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_model(name: str, url: str, min_size: int) -> Path:
    path = _model_root() / name
    if path.is_file() and path.stat().st_size > min_size:
        return path

    tmp = path.with_suffix(".download")
    tmp.unlink(missing_ok=True)
    print(f"[SR] downloading {name}...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    if not tmp.is_file() or tmp.stat().st_size <= min_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"مدل Super-Resolution معتبر دانلود نشد: {name}")
    tmp.replace(path)
    print(f"[SR] model ready: {path}", flush=True)
    return path


def _new_engine(model_path: Path, model: str, scale: int):
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "ماژول OpenCV dnn_superres نصب نیست؛ requirements جدید را نصب کنید."
        )
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model, scale)
    return sr


@lru_cache(maxsize=1)
def _edsr_engine():
    return _new_engine(
        _ensure_model(EDSR_NAME, EDSR_URL, 30_000_000),
        "edsr",
        4,
    )


@lru_cache(maxsize=1)
def _fsrcnn_engine():
    return _new_engine(
        _ensure_model(FSRCNN_NAME, FSRCNN_URL, 10_000),
        "fsrcnn",
        2,
    )


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر ورودی Super-Resolution معتبر نیست.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("ساختار تصویر برای Super-Resolution پشتیبانی نمی‌شود.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError("ساختار کانال‌های تصویر برای Super-Resolution پشتیبانی نمی‌شود.")
    return image


def _document_reference(source: np.ndarray, scale: int) -> np.ndarray:
    """Create a crisp, geometry-faithful reference for text and table edges."""
    height, width = source.shape[:2]
    reference = cv2.resize(
        source,
        (width * scale, height * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )

    lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    denoised = cv2.fastNlMeansDenoising(lightness, None, 4, 7, 21)
    local = cv2.createCLAHE(clipLimit=1.75, tileGridSize=(8, 8)).apply(denoised)
    lightness = cv2.addWeighted(denoised, 0.38, local, 0.62, 0)
    reference = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

    # Multi-scale unsharp mask. Strong enough for screenshots/documents, but bounded.
    blur_small = cv2.GaussianBlur(reference, (0, 0), 0.75)
    detail_small = cv2.addWeighted(reference, 1.72, blur_small, -0.72, 0)
    blur_large = cv2.GaussianBlur(detail_small, (0, 0), 1.8)
    detail_large = cv2.addWeighted(detail_small, 1.18, blur_large, -0.18, 0)
    return detail_large


def _edge_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = max(10.0, float(np.percentile(magnitude, 82.0)))
    mask = np.clip(magnitude / threshold, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), 0.8)


def _hybrid_document_fusion(
    source: np.ndarray,
    sr_image: np.ndarray,
    scale: int,
) -> np.ndarray:
    reference = _document_reference(source, scale)
    mask = _edge_mask(reference)[..., None]

    # Smooth regions use learned SR. Text/table edges favor the crisp reference.
    reference_weight = 0.22 + (0.68 * mask)
    fused = (
        sr_image.astype(np.float32) * (1.0 - reference_weight)
        + reference.astype(np.float32) * reference_weight
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


def _natural_fusion(source: np.ndarray, sr_image: np.ndarray, scale: int) -> np.ndarray:
    height, width = source.shape[:2]
    reference = cv2.resize(
        source,
        (width * scale, height * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )
    mask = _edge_mask(reference)[..., None]
    reference_weight = 0.14 + (0.42 * mask)
    fused = (
        sr_image.astype(np.float32) * (1.0 - reference_weight)
        + reference.astype(np.float32) * reference_weight
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


def _finish(image: np.ndarray, profile: str) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clip = 1.65 if profile in {"سند", "اسکن ضعیف"} else 1.35
    local = cv2.createCLAHE(clipLimit=clip, tileGridSize=(10, 10)).apply(lightness)
    mix = 0.46 if profile in {"سند", "اسکن ضعیف"} else 0.30
    lightness = cv2.addWeighted(lightness, 1.0 - mix, local, mix, 0)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)

    sigma = 0.72 if profile in {"سند", "اسکن ضعیف"} else 0.62
    amount = 0.22 if profile in {"سند", "اسکن ضعیف"} else 0.10
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigma)
    enhanced = cv2.addWeighted(enhanced, 1.0 + amount, blur, -amount, 0)
    return enhanced


def super_resolve_visual(
    image: np.ndarray,
    scale: float = 4.0,
    profile: str = "سند",
) -> np.ndarray:
    """Profile-aware visual SR; OCR remains on its independent text-safe path."""
    image = _ensure_bgr(image)
    requested = max(1.0, float(scale))

    try:
        print(f"[SR] engine=EDSR x4 profile={profile}", flush=True)
        upscaled = _edsr_engine().upsample(image)
        native_scale = 4.0
        if profile in {"سند", "اسکن ضعیف"}:
            upscaled = _hybrid_document_fusion(image, upscaled, 4)
        else:
            upscaled = _natural_fusion(image, upscaled, 4)
    except Exception as exc:
        print(f"[SR] EDSR unavailable, fallback=FSRCNN x2 reason={exc}", flush=True)
        upscaled = _fsrcnn_engine().upsample(image)
        native_scale = 2.0
        if profile in {"سند", "اسکن ضعیف"}:
            upscaled = _hybrid_document_fusion(image, upscaled, 2)
        else:
            upscaled = _natural_fusion(image, upscaled, 2)

    upscaled = _finish(upscaled, profile)

    if abs(requested - native_scale) > 0.01:
        height, width = image.shape[:2]
        target = (
            max(1, round(width * requested)),
            max(1, round(height * requested)),
        )
        interpolation = cv2.INTER_LANCZOS4 if requested > native_scale else cv2.INTER_AREA
        upscaled = cv2.resize(upscaled, target, interpolation=interpolation)

    return upscaled
