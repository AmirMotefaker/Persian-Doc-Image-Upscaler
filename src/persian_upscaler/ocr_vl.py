from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from html.parser import HTMLParser
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .ocr import OCRResult


class _TableTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            cleaned = " ".join(data.split())
            if cleaned:
                self._cell.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            value = " ".join(self._cell).strip()
            self._row.append(value)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _html_table_to_tsv(value: str) -> str:
    if "<table" not in value.lower():
        return ""
    parser = _TableTextParser()
    try:
        parser.feed(value)
    except Exception:
        return ""
    lines = ["\t".join(cell for cell in row if cell) for row in parser.rows]
    return "\n".join(line for line in lines if line.strip())


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
            table_text = _html_table_to_tsv(value)
            return table_text or value.strip()
    return ""


def _canonical_text(result: Any) -> str:
    payload = _result_json(result)
    blocks = _find_parsing_blocks(payload)
    parts = [_block_content(block) for block in blocks]
    parts = [part for part in parts if part]
    if parts:
        return "\n\n".join(parts)

    markdown = _result_markdown(result)
    if markdown:
        table_text = _html_table_to_tsv(markdown)
        return table_text or markdown

    for key in ("text", "content", "markdown"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            table_text = _html_table_to_tsv(value)
            return table_text or value.strip()
    return ""


def _vl_backend() -> str:
    configured = os.environ.get("DAQIQKHAN_VL_BACKEND")
    if configured:
        return configured.strip().lower()
    return "llama-cpp-server" if os.name == "nt" else "native"


def _vl_server_url() -> str:
    return os.environ.get("DAQIQKHAN_VL_SERVER_URL", "http://127.0.0.1:8118/v1").rstrip("/")


@lru_cache(maxsize=1)
def _vl_api_model_name() -> str:
    backend = _vl_backend()
    if backend == "native":
        return ""

    url = _vl_server_url() + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR-VL server is not ready. Start the local llama.cpp VLM service first."
        ) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise RuntimeError("PaddleOCR-VL server returned no model id")

    model_id = str(data[0].get("id") or "").strip()
    if not model_id:
        raise RuntimeError("PaddleOCR-VL server model id is empty")

    print(f"[OCR] VL server model={model_id}", flush=True)
    return model_id


def _require_vl_server() -> None:
    if _vl_backend() != "native":
        _vl_api_model_name()


def _prepare_vl_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر OCR معتبر نیست.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ساختار تصویر OCR-VL پشتیبانی نمی‌شود.")

    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side >= 1100:
        return image.copy()

    scale = min(3.0, 1100.0 / max(1, long_side))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    print(
        f"[OCR] VL deterministic upscale {width}x{height} -> {target[0]}x{target[1]}",
        flush=True,
    )
    return cv2.resize(image, target, interpolation=cv2.INTER_CUBIC)


@lru_cache(maxsize=2)
def _pipeline(use_layout_detection: bool):
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR-VL dependencies are missing. Install paddleocr[doc-parser]==3.7.0."
        ) from exc

    backend = _vl_backend()
    _require_vl_server()
    print(
        "[OCR] engine=PaddleOCR-VL-1.6 "
        f"layout_detection={use_layout_detection} vl_backend={backend}",
        flush=True,
    )

    kwargs: dict[str, Any] = {
        "pipeline_version": "v1.6",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": use_layout_detection,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "format_block_content": True,
        "merge_layout_blocks": True,
        "use_queues": False,
    }
    if backend != "native":
        kwargs["vl_rec_backend"] = backend
        kwargs["vl_rec_server_url"] = _vl_server_url()
        kwargs["vl_rec_api_model_name"] = _vl_api_model_name()
        kwargs["vl_rec_max_concurrency"] = 1

    return PaddleOCRVL(**kwargs)


def recognize_document(
    image: np.ndarray,
    profile: str = "سند",
) -> OCRResult:
    prepared = _prepare_vl_image(image)
    use_layout_detection = profile in {"سند", "اسکرین‌شات", "اسکن ضعیف"}
    started = perf_counter()
    print(
        f"[OCR] PaddleOCR-VL-1.6 start profile={profile} "
        f"layout={use_layout_detection} size={prepared.shape[1]}x{prepared.shape[0]}",
        flush=True,
    )

    outputs = _pipeline(use_layout_detection).predict(prepared)
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
        pass_name=f"paddleocr-vl-1.6:{_vl_backend()}",
        elapsed_seconds=elapsed,
        layout_text=canonical,
    )
