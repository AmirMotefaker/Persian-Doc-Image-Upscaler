import numpy as np

from persian_upscaler import super_resolution


def test_text_profiles_never_call_realesrgan(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("Generative Real-ESRGAN must not run on Persian text profiles")

    monkeypatch.setattr(super_resolution, "_realesrgan_restore", _forbidden)
    image = np.full((32, 64, 3), 220, dtype=np.uint8)

    for profile in ("سند", "اسکرین‌شات", "اسکن ضعیف"):
        output = super_resolution.super_resolve_visual(
            image,
            scale=2.0,
            profile=profile,
        )
        assert output.shape == (128, 256, 3)
        assert output.dtype == np.uint8


def test_document_hd_is_visibly_different_from_plain_resize():
    image = np.full((32, 64, 3), 235, dtype=np.uint8)
    image[10:22, 18:46] = 35

    output = super_resolution.super_resolve_visual(image, scale=4.0, profile="سند")
    plain = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)

    assert output.shape == plain.shape
    assert float(np.mean(np.abs(output.astype(float) - plain.astype(float)))) > 0.1
