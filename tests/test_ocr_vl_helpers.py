import numpy as np

from persian_upscaler.ocr_vl import (
    _looks_like_table,
    _normalize_persian_text,
    _quality_score,
)


def test_persian_normalization_unifies_arabic_codepoints():
    value = _normalize_persian_text("قيمت\tاولين\nأستانه كالا")
    assert value == "قیمت\tاولین\nاستانه کالا"


def test_table_detector_finds_grid_lines():
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    image[20:22, :] = 0
    image[60:62, :] = 0
    image[100:102, :] = 0
    image[:, 40:42] = 0
    image[:, 100:102] = 0
    image[:, 150:152] = 0
    assert _looks_like_table(image)


def test_quality_score_rewards_structured_persian_table():
    structured = "قیمت\tحجم\n27,000,000\t183\nقیمت پایانی\t27,135,086"
    weak = "price 27000000"
    assert _quality_score(structured) > _quality_score(weak)
