from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

TEXT_PROFILES = {"سند", "اسکرین‌شات", "اسکن ضعیف"}


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


def _local_contrast(image: np.ndarray, weak_scan: bool) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=1.60 if weak_scan else 1.45,
        tileGridSize=(8, 8),
    )
    local = clahe.apply(lightness)
    blend = 0.28 if weak_scan else 0.18
    lightness = cv2.addWeighted(lightness, 1.0 - blend, local, blend, 0)
    return cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def _native_text_highboost(image: np.ndarray, weak_scan: bool) -> np.ndarray:
    """Strengthen only source glyph edges before enlargement."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    sigma = 0.78 if weak_scan else 0.65
    blur = cv2.GaussianBlur(lightness, (0, 0), sigma)
    amount = 1.10 if weak_scan else 1.35
    boosted = cv2.addWeighted(lightness, 1.0 + amount, blur, -amount, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blackhat = cv2.morphologyEx(boosted, cv2.MORPH_BLACKHAT, kernel)
    stroke_gain = 0.15 if weak_scan else 0.11
    boosted = np.clip(
        boosted.astype(np.float32) - blackhat.astype(np.float32) * stroke_gain,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        cv2.merge((boosted, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def _edge_limited_unsharp(image: np.ndarray, amount: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = max(10.0, float(np.percentile(magnitude, 70.0)))
    mask = np.clip((magnitude - threshold * 0.20) / max(threshold, 1.0), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.45)[..., None]

    blur = cv2.GaussianBlur(image, (0, 0), 0.45)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)
    mixed = (
        image.astype(np.float32) * (1.0 - mask)
        + sharpened.astype(np.float32) * mask
    )
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _document_restore(image: np.ndarray, scale: int, weak_scan: bool) -> np.ndarray:
    """Deterministic, non-generative Persian document restoration."""
    restored = _local_contrast(image, weak_scan=weak_scan)
    restored = _native_text_highboost(restored, weak_scan=weak_scan)
    height, width = restored.shape[:2]
    output = cv2.resize(
        restored,
        (width * scale, height * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return _edge_limited_unsharp(output, amount=0.68 if weak_scan else 0.58)


def _realesrgan_restore(image: np.ndarray, requested_scale: float) -> np.ndarray:
    """Compatibility seam; text profiles must never call this generative path."""
    height, width = image.shape[:2]
    scale = max(1.0, float(requested_scale))
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_LANCZOS4,
    )


def _natural_restore(image: np.ndarray, requested_scale: float) -> np.ndarray:
    return _realesrgan_restore(image, requested_scale)


def super_resolve_visual(
    image: np.ndarray,
    scale: float = 4.0,
    profile: str = "سند",
) -> np.ndarray:
    started = perf_counter()
    image = _ensure_bgr(image)
    requested = max(1.0, float(scale))

    if profile in TEXT_PROFILES:
        document_scale = 4
        weak_scan = profile == "اسکن ضعیف"
        print(
            f"[SR] engine=PersianDocumentHDv2 x{document_scale} profile={profile} "
            "generative=false",
            flush=True,
        )
        output = _document_restore(image, document_scale, weak_scan=weak_scan)
    else:
        output = _natural_restore(image, requested)

    print(
        f"[SR] done size={output.shape[1]}x{output.shape[0]} "
        f"elapsed={perf_counter() - started:.2f}s",
        flush=True,
    )
    return output
