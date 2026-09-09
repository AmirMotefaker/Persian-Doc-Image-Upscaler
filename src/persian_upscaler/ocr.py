from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
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


@lru_cache(maxsize=2)
def get_ocr(device: str = "cpu") -> PaddleOCR:
    # Persian is supported by PP-OCRv5 multilingual recognition.
    return PaddleOCR(
        lang="fa",
        ocr_version="PP-OCRv5",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def recognize(image: np.ndarray, device: str = "cpu") -> OCRResult:
    results = get_ocr(device).predict(image)
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
