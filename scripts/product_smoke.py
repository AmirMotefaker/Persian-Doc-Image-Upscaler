from pathlib import Path

from app import SAMPLE_SCAN, run_single


def main() -> None:
    comparison, ocr_preview, summary, enhanced, text_file = run_single(
        SAMPLE_SCAN,
        "سند",
        1.5,
        "PNG",
        "fa",
    )

    before, after = comparison
    required = [before, after, ocr_preview, enhanced, text_file]
    missing = [item for item in required if not Path(item).is_file()]
    if missing:
        raise RuntimeError(f"Missing product artifacts: {missing}")
    if not summary.strip():
        raise RuntimeError("OCR summary is empty")

    print("Product callback smoke: PASS")
    print(f"Before: {before}")
    print(f"After: {after}")
    print(f"OCR preview: {ocr_preview}")
    print(f"Enhanced: {enhanced}")
    print(f"Text: {text_file}")


if __name__ == "__main__":
    main()
