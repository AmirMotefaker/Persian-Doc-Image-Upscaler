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
DOCUMENT_PROFILES = {"سند", "اسکن ضعیف"}


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
        raise ValueError("تصویر ورودی معتبر نیست.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("ساختار تصویر پشتیبانی نمی‌شود.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError("ساختار کانال‌های تصویر پشتیبانی نمی‌شود.")
    return image


def _normalize_document_luma(image: np.ndarray, weak_scan: bool) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    if weak_scan:
        local = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8)).apply(lightness)
        lightness = cv2.addWeighted(lightness, 0.74, local, 0.26, 0)
    else:
        local = cv2.createCLAHE(clipLimit=1.10, tileGridSize=(10, 10)).apply(lightness)
        lightness = cv2.addWeighted(lightness, 0.91, local, 0.09, 0)

    return cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def _edge_limited_unsharp(image: np.ndarray, amount: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = max(18.0, float(np.percentile(magnitude, 82.0)))
    mask = np.clip((magnitude - threshold * 0.35) / threshold, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.65)[..., None]

    blur = cv2.GaussianBlur(image, (0, 0), 0.52)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    mixed = (
        image.astype(np.float32) * (1.0 - mask)
        + sharpened.astype(np.float32) * mask
    )
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _document_restore(image: np.ndarray, scale: int, weak_scan: bool) -> np.ndarray:
    """Fidelity-first enlargement for Persian documents.

    No denoising is applied after enlargement: on tiny Persian glyphs that removes
    real stroke information. The path uses mild luminance normalization, Lanczos
    enlargement and edge-gated sharpening only.
    """
    cleaned = _normalize_document_luma(image, weak_scan=weak_scan)
    height, width = cleaned.shape[:2]
    output = cv2.resize(
        cleaned,
        (width * scale, height * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return _edge_limited_unsharp(output, amount=0.20 if weak_scan else 0.16)


def _natural_ai_restore(image: np.ndarray) -> np.ndarray:
    try:
        print("[SR] engine=ESPCN x4 profile=natural", flush=True)
        return _engine().upsample(image)
    except Exception as exc:
        print(f"[SR] ESPCN unavailable; fallback=Lanczos reason={exc}", flush=True)
        height, width = image.shape[:2]
        return cv2.resize(
            image,
            (width * 4, height * 4),
            interpolation=cv2.INTER_LANCZOS4,
        )


def super_resolve_visual(
    image: np.ndarray,
    scale: float = 2.0,
    profile: str = "سند",
) -> np.ndarray:
    started = perf_counter()
    image = _ensure_bgr(image)
    requested = max(1.0, float(scale))

    if profile in DOCUMENT_PROFILES:
        document_scale = max(1, min(2, round(requested)))
        weak_scan = profile == "اسکن ضعیف"
        print(
            f"[SR] engine=DocumentRestoreSharp x{document_scale} profile={profile}",
            flush=True,
        )
        output = _document_restore(image, document_scale, weak_scan=weak_scan)
    else:
        output = _natural_ai_restore(image)
        native_scale = 4
        if abs(requested - native_scale) > 0.01:
            height, width = image.shape[:2]
            target = (
                max(1, round(width * requested)),
                max(1, round(height * requested)),
            )
            interpolation = (
                cv2.INTER_LANCZOS4 if requested > native_scale else cv2.INTER_AREA
            )
            output = cv2.resize(output, target, interpolation=interpolation)

    print(
        f"[SR] done size={output.shape[1]}x{output.shape[0]} "
        f"elapsed={perf_counter() - started:.2f}s",
        flush=True,
    )
    return output
