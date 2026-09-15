from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from persian_upscaler.ocr_vl import recognize_document
from persian_upscaler.restoration.realesrgan_engine import restore_file
from persian_upscaler.restoration.registry import AVAILABLE_ENGINES
from persian_upscaler.super_resolution import super_resolve_visual

FIXTURE = Path(r"C:\Market\گواهی سپرده مس کاتد.png")
OUT_DIR = Path(r"C:\Project\_DAQIQKHAN_P2_WAVE2")
EXPECTED_TOKENS = (
    "27,000,000",
    "27,768,000",
    "27,010,030",
    "26,990,030",
    "27,089,900",
    "26,990,020",
    "12:00:00",
    "28,905,560",
    "23,650,010",
    "55,918",
    "حجم معاملات",
    "قیمت",
)


def read_image_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image_unicode(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_CUBIC)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def normalize_text(text: str) -> str:
    table = str.maketrans(
        {
            "ي": "ی",
            "ى": "ی",
            "ك": "ک",
            "ة": "ه",
            "ۀ": "ه",
            "أ": "ا",
            "إ": "ا",
            "ٱ": "ا",
        }
    )
    return " ".join(text.translate(table).replace("\t", " ").split())


def compact(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def token_recall(text: str) -> dict[str, object]:
    haystack = compact(text)
    found = [token for token in EXPECTED_TOKENS if compact(token) in haystack]
    missing = [token for token in EXPECTED_TOKENS if compact(token) not in haystack]
    return {
        "expected": len(EXPECTED_TOKENS),
        "found": len(found),
        "recall": round(len(found) / len(EXPECTED_TOKENS), 4),
        "found_tokens": found,
        "missing_tokens": missing,
    }


def ocr_candidate(image: np.ndarray, name: str) -> tuple[str, dict[str, object]]:
    print(f"\n[P2] OCR fidelity check start engine={name}", flush=True)
    result = recognize_document(image, profile="سند")
    text = (result.layout_text or result.text).strip()
    stats = token_recall(text)
    print(
        f"[P2] OCR fidelity engine={name} "
        f"tokens={stats['found']}/{stats['expected']} recall={stats['recall']:.1%}",
        flush=True,
    )
    return text, stats


def panel(image: np.ndarray, label: str, target_size: tuple[int, int]) -> np.ndarray:
    rendered = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(rendered, 52, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(
        canvas,
        label,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    report: dict[str, object] = {
        "phase": "p2-wave2-realesrgan",
        "fixture": str(FIXTURE),
        "engines": [
            {
                "key": engine.key,
                "status": engine.status,
                "default_allowed": engine.can_be_default,
            }
            for engine in AVAILABLE_ENGINES
        ],
    }

    if not FIXTURE.exists():
        report["status"] = "fixture-missing"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    executable_raw = os.environ.get("DAQIQKHAN_REALESRGAN_BIN", "").strip()
    model_dir_raw = os.environ.get("DAQIQKHAN_REALESRGAN_MODEL_DIR", "").strip()
    if not executable_raw or not model_dir_raw:
        raise RuntimeError(
            "Real-ESRGAN benchmark environment is not configured. "
            "Run scripts/run-p2-wave2.ps1."
        )

    executable = Path(executable_raw)
    model_dir = Path(model_dir_raw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in OUT_DIR.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    source = read_image_unicode(FIXTURE)
    source_h, source_w = source.shape[:2]

    print("\n=== P2 WAVE 2 — BASELINE ===", flush=True)
    baseline = super_resolve_visual(source, scale=4.0, profile="سند")
    baseline_path = OUT_DIR / "02-baseline-persian-document-hdv2.png"
    write_image_unicode(baseline_path, baseline)

    print("\n=== P2 WAVE 2 — REAL-ESRGAN CANDIDATE ===", flush=True)
    candidate_path = OUT_DIR / "03-candidate-realesrgan-x4plus.png"
    restore_file(
        FIXTURE,
        candidate_path,
        executable,
        model_dir,
        model_name="realesrgan-x4plus",
        scale=4,
    )
    candidate = read_image_unicode(candidate_path)

    write_image_unicode(OUT_DIR / "01-original.png", source)

    print("\n=== P2 WAVE 2 — OCR FIDELITY ===", flush=True)
    baseline_text, baseline_tokens = ocr_candidate(baseline, "persian-document-hdv2")
    candidate_text, candidate_tokens = ocr_candidate(candidate, "realesrgan-x4plus")

    (OUT_DIR / "04-baseline-ocr.txt").write_text(baseline_text, encoding="utf-8")
    (OUT_DIR / "05-realesrgan-ocr.txt").write_text(candidate_text, encoding="utf-8")

    target_size = (baseline.shape[1], baseline.shape[0])
    montage = np.hstack(
        [
            panel(source, "ORIGINAL (BICUBIC VIEW)", target_size),
            panel(baseline, "PERSIAN DOCUMENT HDV2", target_size),
            panel(candidate, "REAL-ESRGAN X4PLUS", target_size),
        ]
    )
    montage_path = OUT_DIR / "06-montage.png"
    write_image_unicode(montage_path, montage)

    candidate_resized = candidate
    if candidate.shape != baseline.shape:
        candidate_resized = cv2.resize(
            candidate,
            target_size,
            interpolation=cv2.INTER_CUBIC,
        )

    baseline_recall = float(baseline_tokens["recall"])
    candidate_recall = float(candidate_tokens["recall"])
    if candidate_recall < baseline_recall:
        decision = "reject-ocr-fidelity-regression"
    else:
        decision = "needs-human-visual-review"

    report.update(
        {
            "status": "benchmark-complete",
            "source_size": [source_w, source_h],
            "baseline_size": [baseline.shape[1], baseline.shape[0]],
            "candidate_size": [candidate.shape[1], candidate.shape[0]],
            "baseline": {
                "sharpness": round(sharpness(baseline), 4),
                "contrast": round(contrast(baseline), 4),
                "ocr_tokens": baseline_tokens,
            },
            "realesrgan": {
                "model": "realesrgan-x4plus",
                "sharpness": round(sharpness(candidate), 4),
                "contrast": round(contrast(candidate), 4),
                "mad_vs_baseline": round(mean_abs_diff(candidate_resized, baseline), 4),
                "ocr_tokens": candidate_tokens,
            },
            "decision": decision,
            "safety": {
                "product_default_changed": False,
                "service_integration_changed": False,
                "requires_human_visual_review": True,
            },
            "artifacts": {
                "directory": str(OUT_DIR),
                "montage": str(montage_path),
            },
        }
    )

    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== P2 WAVE 2 REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {OUT_DIR}")
    print(f"Decision: {decision}")

    if decision.startswith("reject-"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
