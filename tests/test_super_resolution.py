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
        assert output.shape == (64, 128, 3)
        assert output.dtype == np.uint8
