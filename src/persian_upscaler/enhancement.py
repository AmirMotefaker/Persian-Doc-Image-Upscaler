import cv2
import numpy as np

PROFILES = ("طبیعی", "سند", "اسکن ضعیف")


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
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def restore_visual(image: np.ndarray, profile: str = "طبیعی", scale: float = 2.0) -> np.ndarray:
    """Non-generative enhancement intended to preserve Persian glyph geometry."""
    image = resize_for_text(image, scale=scale)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)

    clip = 1.8 if profile == "طبیعی" else 2.4 if profile == "سند" else 3.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )

    if profile == "اسکن ضعیف":
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 4, 4, 7, 21)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    amount = 1.15 if profile == "طبیعی" else 1.35
    return cv2.addWeighted(enhanced, amount, blurred, -(amount - 1.0), 0)


def prepare_for_ocr(restored: np.ndarray, profile: str = "طبیعی") -> np.ndarray:
    """Create a dedicated OCR image without altering the user's visual output."""
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
