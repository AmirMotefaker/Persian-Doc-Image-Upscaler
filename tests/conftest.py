import sys
import types
from importlib import util

if util.find_spec("paddleocr") is None:
    paddleocr = types.ModuleType("paddleocr")

    class _UnavailablePaddleRuntime:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PaddleOCR runtime is not installed in lightweight CI")

    paddleocr.PaddleOCR = _UnavailablePaddleRuntime
    paddleocr.TextRecognition = _UnavailablePaddleRuntime
    sys.modules["paddleocr"] = paddleocr
