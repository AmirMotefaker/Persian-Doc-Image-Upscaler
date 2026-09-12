from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from .ocr import OCRResult


def _result_json(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", None)
    value = value() if callable(value) else value
    if isinstance(value, dict):
        return value
    return {}


def _result_markdown(result: Any) -> str:
    value = getattr(result, "markdown", None)
    value = value() if callable(value) else value
    if not isinstance(value, dict):
        return ""

    for key in ("markdown_texts", "text"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, (list, tuple)):
            parts = [str(item).strip() for item in candidate if str(item).strip()]
            if parts:
                return "\n\n".join(parts)
    return ""


def _find_parsing_blocks(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        for key in ("parsing_res_list", "parsingResList"):
            value = node.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in node.values():
            found = _find_parsing_blocks(value)
            if found:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _find_parsing_blocks(value)
            if found:
                return found
    return []


def _block_content(block: dict[str, Any]) -> str:
    for key in ("block_content", "blockContent", "content", "text"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _canonical_text(result: Any) -> str:
    markdown = _result_markdown(result)
    if markdown:
        return markdown

    payload = _result_json(result)
    blocks = _find_parsing_blocks(payload)
    parts = [_block_content(block) for block in blocks]
    parts = [part for part in parts if part]
    if parts:
        return "\n\n".join(parts)

    for key in ("text", "content", "markdown"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


@lru_cache(maxsize=2)
def _pipeline(use_layout_detection: bool):
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR-VL dependencies are missing. Install paddleocr[doc-parser]==3.7.0."
        ) from exc

    print(
        "[OCR] engine=PaddleOCR-VL-1.6 "
        f"layout_detection={use_layout_detection} backend=native",
        flush=True,
    )
    return PaddleOCRVL(
        pipeline_version="v1.6",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=use_layout_detection,
        use_chart_recognition=False,
        use_seal_recognition=False,
        format_block_content=True,
        merge_layout_blocks=True,
    )


def recognize_document(
    image: np.ndarray,
    profile: str = "سند",
) -> OCRResult:
    if image is None or image.size == 0:
        raise ValueError("تصویر OCR معتبر نیست.")

    use_layout_detection = profile in {"سند", "اسکن ضعیف"}
    started = perf_counter()
    print(
        f"[OCR] PaddleOCR-VL-1.6 start profile={profile} "
        f"layout={use_layout_detection} size={image.shape[1]}x{image.shape[0]}",
        flush=True,
    )

    outputs = _pipeline(use_layout_detection).predict(image)
    texts: list[str] = []
    for result in outputs:
        text = _canonical_text(result)
        if text:
            texts.append(text)

    canonical = "\n\n".join(texts).strip()
    if not canonical:
        raise RuntimeError("PaddleOCR-VL-1.6 returned no usable Persian document text")

    lines = tuple((line, 0.0) for line in canonical.splitlines() if line.strip())
    elapsed = perf_counter() - started
    print(
        f"[OCR] PaddleOCR-VL-1.6 done lines={len(lines)} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return OCRResult(
        text=canonical,
        average_confidence=0.0,
        lines=lines,
        pass_name="paddleocr-vl-1.6",
        elapsed_seconds=elapsed,
        layout_text=canonical,
    )
