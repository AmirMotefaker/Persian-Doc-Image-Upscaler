import cv2
import numpy as np

PROFILES = ("طبیعی", "سند", "اسکن ضعیف")
ENGINES = ("Text-Safe Pro", "Standard")


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر معتبر نیست.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("ساختار کانال‌های تصویر پشتیبانی نمی‌شود.")
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels != 3:
        raise ValueError(f"تعداد کانال‌های تصویر پشتیبانی نمی‌شود: {channels}")
    return image


def resize_for_text(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    image = _ensure_bgr(image)
    if scale <= 1.0:
        return image.copy()
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LANCZOS4,
    )


def _standard_restore(image: np.ndarray, profile: str) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clip = 1.8 if profile == "طبیعی" else 2.4 if profile == "سند" else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab = lab.copy()
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if profile == "اسکن ضعیف":
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 4, 4, 7, 21)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.9)
    amount = 1.14 if profile == "طبیعی" else 1.32 if profile == "سند" else 1.42
    sharpened = cv2.addWeighted(enhanced, amount, blurred, -(amount - 1.0), 0)
    return cv2.bilateralFilter(sharpened, 3, 18, 18)


def _text_safe_pro_restore(image: np.ndarray, profile: str) -> np.ndarray:
    """High-detail non-generative document restoration tuned for Persian glyphs."""
    base = _standard_restore(image, profile)

    # Work on luminance only so colour edges and UI elements are preserved.
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    # Remove broad illumination variation common in scans/screenshots without
    # eroding Persian dots, teeth or thin stems.
    background = cv2.GaussianBlur(lightness, (0, 0), 11.0)
    corrected = cv2.addWeighted(lightness, 1.35, background, -0.35, 0)

    # Local contrast at two scales improves small text while avoiding halos.
    clahe_small = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(16, 16))
    clahe_large = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
    detail_small = clahe_small.apply(corrected)
    detail_large = clahe_large.apply(corrected)
    fused = cv2.addWeighted(detail_small, 0.62, detail_large, 0.38, 0)

    # Edge-limited sharpening: only strengthen genuine text edges.
    soft = cv2.GaussianBlur(fused, (0, 0), 0.72)
    high = cv2.subtract(fused, soft)
    edge_mask = cv2.Canny(fused, 42, 118)
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 0.8).astype(np.float32) / 255.0
    high_f = high.astype(np.float32)
    fused_f = fused.astype(np.float32)
    sharpened = np.clip(fused_f + (high_f * edge_mask * 1.45), 0, 255).astype(np.uint8)

    # Restore tiny isolated dots that aggressive denoise can otherwise weaken.
    blackhat = cv2.morphologyEx(
        sharpened,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    sharpened = cv2.subtract(sharpened, cv2.multiply(blackhat, 0.18))

    lab[:, :, 0] = sharpened
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(result, 3, 12, 12)


def restore_visual(
    image: np.ndarray,
    profile: str = "طبیعی",
    scale: float = 2.0,
    engine: str = "Text-Safe Pro",
) -> np.ndarray:
    """Restore Persian text while keeping glyph geometry non-generative."""
    image = resize_for_text(image, scale=scale)
    if engine == "Standard":
        return _standard_restore(image, profile)
    if engine != "Text-Safe Pro":
        raise ValueError(f"موتور پردازش ناشناخته است: {engine}")
    return _text_safe_pro_restore(image, profile)


def prepare_for_ocr(restored: np.ndarray, profile: str = "طبیعی") -> np.ndarray:
    restored = _ensure_bgr(restored)
    gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)

    if profile == "طبیعی":
        return gray

    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    block_size = 31 if profile == "سند" else 41
    c_value = 9 if profile == "سند" else 12
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def build_ocr_candidates(
    restored: np.ndarray,
    profile: str = "طبیعی",
) -> list[tuple[str, np.ndarray]]:
    restored = _ensure_bgr(restored)
    gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    contrast = cv2.bilateralFilter(contrast, 5, 28, 28)

    if profile == "طبیعی":
        return [("restored-color", restored)]

    adaptive = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31 if profile == "سند" else 41,
        9 if profile == "سند" else 12,
    )

    if profile == "سند":
        return [("restored-color", restored), ("adaptive", adaptive)]

    _, otsu = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return [
        ("contrast-gray", contrast),
        ("adaptive", adaptive),
        ("otsu", otsu),
    ]
