import os
import sys
import types

# Unit tests must never initialize the real Paddle/PaddleOCR runtime.  The real
# document parser is verified separately by the runtime smoke command before the
# app starts.  This keeps pytest deterministic on Windows and in lightweight CI.
if os.environ.get("DAQIQKHAN_TEST_REAL_PADDLE") != "1":
    paddleocr = types.ModuleType("paddleocr")

    class _UnavailablePaddleRuntime:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PaddleOCR runtime is disabled during unit tests")

    paddleocr.PaddleOCR = _UnavailablePaddleRuntime
    paddleocr.TextDetection = _UnavailablePaddleRuntime
    paddleocr.TextRecognition = _UnavailablePaddleRuntime
    paddleocr.PaddleOCRVL = _UnavailablePaddleRuntime
    sys.modules["paddleocr"] = paddleocr
