import numpy as np

from persian_upscaler.enhancement import prepare_for_ocr, restore_visual


def test_restore_visual_preserves_three_channels():
    image = np.full((40, 80, 3), 240, dtype=np.uint8)
    output = restore_visual(image, profile="سند", scale=2.0)
    assert output.shape == (80, 160, 3)
    assert output.dtype == np.uint8


def test_ocr_preprocessing_is_separate_grayscale_image():
    image = np.full((40, 80, 3), 240, dtype=np.uint8)
    restored = restore_visual(image, profile="سند", scale=1.0)
    ocr_image = prepare_for_ocr(restored, profile="سند")
    assert ocr_image.ndim == 2
    assert restored.ndim == 3


def test_restore_visual_accepts_grayscale_input():
    image = np.full((40, 80), 220, dtype=np.uint8)
    output = restore_visual(image, profile="سند", scale=1.0)
    assert output.shape == (40, 80, 3)


def test_restore_visual_accepts_single_channel_input():
    image = np.full((40, 80, 1), 220, dtype=np.uint8)
    output = restore_visual(image, profile="سند", scale=1.0)
    assert output.shape == (40, 80, 3)


def test_restore_visual_accepts_alpha_input():
    image = np.full((40, 80, 4), 220, dtype=np.uint8)
    output = restore_visual(image, profile="سند", scale=1.0)
    assert output.shape == (40, 80, 3)
