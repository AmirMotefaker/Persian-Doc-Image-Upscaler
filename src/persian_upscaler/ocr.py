from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

# PaddlePaddle 3.3.x can fail on Windows CPU when PIR and oneDNN are combined
# for some OCR graphs. These flags must be set before importing PaddleOCR.
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR


@dataclass(frozen=True)
class OCRResult:
    text: str
    average_confidence: float
    lines: tuple[tuple[str, float], ...]


def _find_payload(node: Any) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if "rec_texts" in node and "rec_scores" in node:
            return node
        for value in node.values():
            found = _find_payload(value)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _find_payload(value)
            if found is not None:
                return found
    return None


def _normalize_inference_image(image: np.ndarray) -> np.ndarray:
    """PaddleOCR detection expects an HxWxC image even after OCR preprocessing."""
    if image is None or image.size == 0:
        raise ValueError("تصویر OCR معتبر نیست.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError("ساختار تصویر OCR پشتیبانی نمی‌شود.")
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if channels != 3:
        raise ValueError(f"تعداد کانال‌های OCR پشتیبانی نمی‌شود: {channels}")
    return image


@lru_cache(maxsize=2)
def get_ocr(device: str = "cpu") -> PaddleOCR:
    return PaddleOCR(
        lang="fa",
        ocr_version="PP-OCRv5",
        device=device,
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def recognize(image: np.ndarray, device: str = "cpu") -> OCRResult:
    inference_image = _normalize_inference_image(image)
    results = get_ocr(device).predict(inference_image)
    lines: list[tuple[str, float]] = []

    for result in results:
        payload = _find_payload(result.json)
        if not payload:
            continue
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        for index, text in enumerate(texts):
            cleaned = str(text).strip()
            if not cleaned:
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            lines.append((cleaned, score))

    average = sum(score for _, score in lines) / len(lines) if lines else 0.0
    return OCRResult(
        text="\n".join(text for text, _ in lines),
        average_confidence=average,
        lines=tuple(lines),
    )
