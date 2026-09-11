from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

BINA_REVISION = "2af6ae7eeb38d195d4d26b13467fccf0e27f37e4"
BINA_BASE_URL = "https://huggingface.co/Reza2kn/Bina-0.2-Rizeh/resolve/" f"{BINA_REVISION}"
BINA_RECOGNIZER_FILES = {
    "inference.json": "b802ea6f182ffb341716b2f649b8f48a2c1f3e4fb6db26f4cfd31771d80cd8b8",
    "inference.pdiparams": "f8e1b5b4e9ef11b46c521fed7d1c9b7e7b2d08c4c879aa01353acd7308a6e9fb",
    "inference.yml": "650bfd52635c3a5680479df99a4c5cb3b94e2875b25a7bc3aea45ec1bad6a841",
}
BINA_DETECTOR_FILES = {
    "inference.json": "0f1a7ec35da36173529c7a60238b7f7919e3831929c3f700ad90ad4896adecd5",
    "inference.pdiparams": "85218d2e3d98f5a21c58b4220627be923a97aee5db3cc71f39536ab31ac53960",
    "inference.yml": "7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871",
}
MAX_OCR_PIXELS = 2_000_000
MAX_OCR_SIDE = 1800
_LTR_RUN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :*./%+-")


@dataclass(frozen=True)
class OCRResult:
    text: str
    average_confidence: float
    lines: tuple[tuple[str, float], ...]
    pass_name: str = "bina"
    elapsed_seconds: float = 0.0
    layout_text: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_root() -> Path:
    root = Path(
        os.environ.get(
            "DAQIQKHAN_BINA_MODEL_DIR",
            Path.home() / ".cache" / "daqiqkhan" / "bina-0.2-rizeh",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_component(component: str, files: dict[str, str]) -> Path:
    root = _runtime_root() / component
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha in files.items():
        target = root / filename
        if target.is_file() and _sha256(target) == expected_sha:
            continue

        tmp = target.with_suffix(target.suffix + ".download")
        tmp.unlink(missing_ok=True)
        print(f"[OCR] Bina download start component={component} file={filename}", flush=True)
        urllib.request.urlretrieve(
            f"{BINA_BASE_URL}/{component}/{filename}?download=true",
            tmp,
        )
        if not tmp.is_file() or _sha256(tmp) != expected_sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Bina model checksum failed: {component}/{filename}")
        tmp.replace(target)
        print(f"[OCR] Bina model ready component={component} file={filename}", flush=True)
    return root


def _ensure_bina_runtime() -> tuple[Path, Path]:
    recognizer = _ensure_component("inference", BINA_RECOGNIZER_FILES)
    detector = _ensure_component("detector", BINA_DETECTOR_FILES)
    return recognizer, detector


def _pred_reverse(text: str) -> str:
    """Convert Bina/Paddle visual-order Persian output to logical reading order."""
    segments: list[str] = []
    current_ltr = ""
    for character in text:
        if character in _LTR_RUN:
            current_ltr += character
            continue
        if current_ltr:
            segments.append(current_ltr)
            current_ltr = ""
        segments.append(character)
    if current_ltr:
        segments.append(current_ltr)
    return "".join(reversed(segments))


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


def _normalize_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("تصویر OCR معتبر نیست.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ساختار تصویر OCR پشتیبانی نمی‌شود.")

    height, width = image.shape[:2]
    side_scale = min(1.0, MAX_OCR_SIDE / max(height, width))
    pixel_scale = min(1.0, (MAX_OCR_PIXELS / max(1, height * width)) ** 0.5)
    scale = min(side_scale, pixel_scale)
    if scale < 0.999:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _boxes(payload: dict[str, Any], count: int) -> np.ndarray | None:
    raw = payload.get("rec_boxes")
    if raw is None:
        return None
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 4:
        return None
    return array[:, :4]


def _group_rows(array: np.ndarray) -> list[list[int]]:
    if len(array) == 0:
        return []
    heights = np.maximum(1.0, array[:, 3] - array[:, 1])
    tolerance = max(6.0, float(np.median(heights)) * 0.55)
    items = sorted(
        (((box[1] + box[3]) / 2.0, index) for index, box in enumerate(array)),
        key=lambda item: item[0],
    )
    rows: list[list[int]] = []
    centers: list[float] = []
    for center_y, index in items:
        if not rows or abs(center_y - centers[-1]) > tolerance:
            rows.append([index])
            centers.append(float(center_y))
            continue
        rows[-1].append(index)
        centers[-1] = float(
            np.mean([(array[i, 1] + array[i, 3]) / 2.0 for i in rows[-1]])
        )
    return rows


def _render_layout(
    payload: dict[str, Any],
    texts: list[str],
    scores: list[float],
) -> tuple[str, tuple[tuple[str, float], ...]]:
    array = _boxes(payload, len(texts))
    if array is None:
        lines = tuple(
            (text.strip(), scores[index])
            for index, text in enumerate(texts)
            if text.strip()
        )
        return "\n".join(text for text, _ in lines), lines

    rendered: list[str] = []
    line_items: list[tuple[str, float]] = []
    for row_indices in _group_rows(array):
        row_indices.sort(key=lambda index: -float((array[index, 0] + array[index, 2]) / 2.0))
        cells = [texts[index].strip() for index in row_indices]
        valid = [(index, cell) for index, cell in zip(row_indices, cells, strict=True) if cell]
        if not valid:
            continue
        row_text = "\t".join(cell for _, cell in valid)
        row_score = float(np.mean([scores[index] for index, _ in valid]))
        rendered.append(row_text)
        line_items.append((row_text, row_score))
    return "\n".join(rendered), tuple(line_items)


@lru_cache(maxsize=1)
def get_bina_ocr(device: str = "cpu") -> PaddleOCR:
    recognizer_dir, detector_dir = _ensure_bina_runtime()
    print(
        f"[OCR] engine=Bina-0.2-Rizeh detector=PP-OCRv6-medium device={device}",
        flush=True,
    )
    return PaddleOCR(
        device=device,
        text_detection_model_dir=str(detector_dir),
        text_recognition_model_dir=str(recognizer_dir),
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=0.0,
    )


def recognize(
    image: np.ndarray,
    device: str = "cpu",
    pass_name: str = "restored-color",
) -> OCRResult:
    started = perf_counter()
    inference_image = _normalize_image(image)
    height, width = inference_image.shape[:2]
    print(f"[OCR] Bina start pass={pass_name} size={width}x{height}", flush=True)

    layout_parts: list[str] = []
    all_lines: list[tuple[str, float]] = []
    for result in get_bina_ocr(device).predict(inference_image):
        payload = _find_payload(result.json)
        if not payload:
            continue
        visual_texts = [str(value) for value in (payload.get("rec_texts") or [])]
        texts = [_pred_reverse(text).strip() for text in visual_texts]
        raw_scores = payload.get("rec_scores") or []
        scores = [
            float(raw_scores[index]) if index < len(raw_scores) else 0.0
            for index in range(len(texts))
        ]
        layout_text, lines = _render_layout(payload, texts, scores)
        if layout_text:
            layout_parts.append(layout_text)
        all_lines.extend(lines)

    average = sum(score for _, score in all_lines) / len(all_lines) if all_lines else 0.0
    layout_text = "\n".join(layout_parts).strip()
    elapsed = perf_counter() - started
    print(
        f"[OCR] Bina done pass={pass_name} lines={len(all_lines)} "
        f"confidence={average:.4f} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return OCRResult(
        text=layout_text,
        average_confidence=average,
        lines=tuple(all_lines),
        pass_name=f"bina:{pass_name}",
        elapsed_seconds=elapsed,
        layout_text=layout_text,
    )


def recognize_best(
    candidates: list[tuple[str, np.ndarray]],
    device: str = "cpu",
) -> OCRResult:
    if not candidates:
        raise ValueError("هیچ ورودی OCR برای ارزیابی وجود ندارد.")

    total_started = perf_counter()
    pass_name, image = candidates[0]
    result = recognize(image, device=device, pass_name=pass_name)
    print(
        f"[OCR] selected engine={result.pass_name} single-pass=true "
        f"total={perf_counter() - total_started:.2f}s",
        flush=True,
    )
    return result
