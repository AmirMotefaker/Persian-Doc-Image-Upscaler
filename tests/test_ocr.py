import numpy as np

from persian_upscaler import ocr


class _FakeResult:
    json = {"res": {"rec_texts": ["سلام دنیا"], "rec_scores": [0.97]}}


class _FakeOCR:
    def __init__(self):
        self.seen_shape = None

    def predict(self, image):
        self.seen_shape = image.shape
        return [_FakeResult()]


def test_recognize_converts_grayscale_to_three_channels(monkeypatch):
    fake = _FakeOCR()
    monkeypatch.setattr(ocr, "get_ocr", lambda device="cpu": fake)

    result = ocr.recognize(np.full((64, 128), 255, dtype=np.uint8))

    assert fake.seen_shape == (64, 128, 3)
    assert result.text == "سلام دنیا"
    assert result.average_confidence == 0.97
