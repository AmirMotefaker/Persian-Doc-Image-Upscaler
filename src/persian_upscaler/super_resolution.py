from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ESPCN_URL = (
    "https://raw.githubusercontent.com/Saafke/ESPCN_Tensorflow/"
    "master/models/ESPCN_x4.pb"
)
ESPCN_NAME = "ESPCN_x4.pb"


def _model_root() -> Path:
    root = Path(
        os.environ.get(
            "DAQIQKHAN_MODEL_DIR",
            Path.home() / ".cache" / "daqiqkhan" / "models",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_model() -> Path:
    path = _model_root() / ESPCN_NAME
    if path.is_file() and path.stat().st_size > 10_000:
        return path

    tmp = path.with_suffix(".download")
    tmp.unlink(missing_ok=True)
    print(f"[SR] downloading {ESPCN_NAME}...", flush=True)
    urllib.request.urlretrieve(ESPCN_URL, tmp)
    if not tmp.is_file() or tmp.stat().st_size <= 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("مدل Lightweight Super-Resolution معتبر دانلود نشد.")
    tmp.replace(path)
    print(f"[SR] model ready: {path}", flush=True)
    return path


@lru_cache(maxsize=1)
def _engine():
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError("ماژول OpenCV dnn_superres در دسترس نیست.")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(_ensure_model()))
    sr.setModel("espcn", 4)
    return sr


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
        raise ValueError("ساختار کانال‌های تصویر پشتیبانی نمی‌شود.")
    return image


def _preclean_document(image: np.ndarray, profile: str) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if profile in {"سند", "اسکن ضعیف"}:
        l = cv2.fastNlMeansDenoising(l, None, 3, 7, 17)
        clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
        local = clahe.apply(l)
        l = cv2.addWeighted(l, 0.45, local, 0.55, 0)
    else:
        clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
        local = clahe.apply(l)
        l = cv2.addWeighted(l, 0.72, local, 0.28, 0)

    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _crisp_reference(image: np.ndarray, scale: int) -> np.ndarray:
    h, w = image.shape[:2]
    reference = cv2.resize(
        image,
        (w * scale, h * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )

    blur1 = cv2.GaussianBlur(reference, (0, 0), 0.55)
    sharp1 = cv2.addWeighted(reference, 1.55, blur1, -0.55, 0)
    blur2 = cv2.GaussianBlur(sharp1, (0, 0), 1.35)
    return cv2.addWeighted(sharp1, 1.10, blur2, -0.10, 0)


def _edge_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    denom = max(12.0, float(np.percentile(magnitude, 84.0)))
    mask = np.clip(magnitude / denom, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), 0.65)


def _fuse_document(
    source: np.ndarray,
    learned: np.ndarray,
    profile: str,
) -> np.ndarray:
    reference = _crisp_reference(source, 4)
    mask = _edge_mask(reference)[..., None]

    if profile in {"سند", "اسکن ضعیف"}:
        # Persian glyph geometry and table edges prefer the deterministic path.
        reference_weight = 0.35 + 0.58 * mask
    else:
        reference_weight = 0.18 + 0.35 * mask

    fused = (
        learned.astype(np.float32) * (1.0 - reference_weight)
        + reference.astype(np.float32) * reference_weight
    )
    return np.clip(fused, 0, 255).astype(np.uint8)


def _finalize(image: np.ndarray, profile: str) -> np.ndarray:
    if profile in {"سند", "اسکن ضعیف"}:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        local = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(12, 12)).apply(l)
        l = cv2.addWeighted(l, 0.70, local, 0.30, 0)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(image, (0, 0), 0.48)
        image = cv2.addWeighted(image, 1.20, blur, -0.20, 0)

    return image


def _fast_fallback(image: np.ndarray, profile: str) -> np.ndarray:
    reference = _crisp_reference(image, 4)
    return _finalize(reference, profile)


def super_resolve_visual(
    image: np.ndarray,
    scale: float = 4.0,
    profile: str = "سند",
) -> np.ndarray:
    """Fast interactive SR for Persian documents; OCR stays on its own path."""
    started = perf_counter()
    image = _ensure_bgr(image)
    cleaned = _preclean_document(image, profile)

    try:
        print(f"[SR] engine=ESPCN x4 profile={profile}", flush=True)
        learned = _engine().upsample(cleaned)
        output = _fuse_document(cleaned, learned, profile)
    except Exception as exc:
        print(f"[SR] lightweight AI unavailable; fast fallback reason={exc}", flush=True)
        output = _fast_fallback(cleaned, profile)

    output = _finalize(output, profile)

    requested = max(1.0, float(scale))
    if abs(requested - 4.0) > 0.01:
        h, w = image.shape[:2]
        target = (max(1, round(w * requested)), max(1, round(h * requested)))
        interpolation = cv2.INTER_LANCZOS4 if requested > 4.0 else cv2.INTER_AREA
        output = cv2.resize(output, target, interpolation=interpolation)

    print(
        f"[SR] done size={output.shape[1]}x{output.shape[0]} "
        f"elapsed={perf_counter() - started:.2f}s",
        flush=True,
    )
    return output
