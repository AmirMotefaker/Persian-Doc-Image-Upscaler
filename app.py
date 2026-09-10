from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from persian_upscaler.service import process_image

PROFILE_CHOICES = {
    "fa": [("سند", "سند"), ("اسکرین‌شات", "طبیعی"), ("اسکن ضعیف", "اسکن ضعیف")],
    "en": [("Document", "سند"), ("Screenshot", "طبیعی"), ("Weak scan", "اسکن ضعیف")],
}
ENGINE_CHOICES = [
    ("Text-Safe Pro", "Text-Safe Pro"),
    ("Standard", "Standard"),
]

TEXT = {
    "fa": {
        "brand": "دقیق‌خوان",
        "tagline": "بهبود تصویر و OCR تخصصی فارسی",
        "ready": "موتور فارسی آماده",
        "headline": "متن فارسی را واضح‌تر کنید",
        "sub": "تصویر را بارگذاری کنید؛ کیفیت متن و خوانایی را بدون تغییر شکل حروف افزایش می‌دهیم.",
        "upload": "تصویر را اینجا رها کنید یا کلیک کنید",
        "sample": "استفاده از نمونه اسکن",
        "profile": "نوع تصویر",
        "engine": "کیفیت پردازش",
        "format": "فرمت خروجی",
        "scale": "افزایش ابعاد",
        "process": "بهبود تصویر و استخراج متن",
        "compare": "قبل / بعد",
        "text": "متن استخراج‌شده",
        "download": "دریافت تصویر",
        "download_text": "دریافت متن",
        "empty": "پس از پردازش، نتیجه در این بخش نمایش داده می‌شود.",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian Image Enhancement & OCR",
        "ready": "Persian engine ready",
        "headline": "Make Persian text clearer",
        "sub": "Upload an image to improve readability while preserving Persian glyph geometry.",
        "upload": "Drop an image here or click to browse",
        "sample": "Use scan sample",
        "profile": "Image type",
        "engine": "Enhancement quality",
        "format": "Output format",
        "scale": "Upscale factor",
        "process": "Enhance image & extract text",
        "compare": "Before / After",
        "text": "Extracted text",
        "download": "Download image",
        "download_text": "Download text",
        "empty": "Your processed result will appear here.",
    },
}


def _header(language: str) -> str:
    t = TEXT[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='topbar' dir='{direction}'>"
        "<div class='brand-wrap'><div class='logo'>د</div><div>"
        f"<div class='brand-name'>{t['brand']}</div>"
        f"<div class='brand-tag'>{t['tagline']}</div></div></div>"
        f"<div class='ready'><span></span>{t['ready']}</div>"
        "</div>"
    )


def _hero(language: str) -> str:
    t = TEXT[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='hero' dir='{direction}'>"
        f"<h1>{t['headline']}</h1><p>{t['sub']}</p>"
        "</div>"
    )


def _comparison_pair(original_path: str, enhanced_path: str) -> tuple[str, str]:
    workdir = Path(tempfile.mkdtemp(prefix="daqiqkhan-compare-"))
    before_path = workdir / "before.png"
    with Image.open(enhanced_path) as enhanced:
        target_size = enhanced.size
    with Image.open(original_path) as original:
        original = original.convert("RGB")
        if original.size != target_size:
            original = original.resize(target_size, Image.Resampling.LANCZOS)
        original.save(before_path, format="PNG")
    return str(before_path), enhanced_path


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_right_aligned(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((max(40, x - width), y), text, fill=fill, font=font)


def _make_scan_sample() -> str:
    workdir = Path(tempfile.mkdtemp(prefix="daqiqkhan-sample-"))
    path = workdir / "persian-scan-sample.png"
    canvas = Image.new("L", (1180, 760), 242)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(38)
    text_font = _font(27)
    lines = [
        "نمونه سند فارسی برای ارزیابی کیفیت OCR",
        "شماره سند: ۱۴۰۵-۰۶-۱۸     تاریخ ثبت: ۱۴۰۵/۰۶/۱۸",
        "نام کالا: گواهی سپرده کالایی",
        "مقدار: ۲۷٬۰۰۰٬۰۰۰ ریال",
        "توضیحات: این تصویر شبیه یک اسکن کم‌کیفیت ساخته شده است.",
        "هدف: بررسی حفظ نقطه‌ها، اعداد و فاصله‌های فارسی.",
    ]
    y = 85
    for index, line in enumerate(lines):
        font = title_font if index == 0 else text_font
        _draw_right_aligned(draw, (1080, y), line, fill=28, font=font)
        y += 92 if index == 0 else 78
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.55))
    canvas = canvas.rotate(
        0.35,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=245,
    )
    canvas.save(path, format="PNG", optimize=True)
    return str(path)


SAMPLE_SCAN = _make_scan_sample()


def run_single(
    image_path,
    profile,
    engine,
    scale,
    output_format,
    language,
    progress=gr.Progress(),
):
    if not image_path:
        raise gr.Error(
            "Please upload an image containing Persian text."
            if language == "en"
            else "لطفاً یک تصویر دارای متن فارسی بارگذاری کنید."
        )

    progress(0.05, desc="Preparing image")
    try:
        enhanced, _ocr_preview, summary, text_file = process_image(
            str(image_path),
            profile=profile,
            scale=float(scale),
            language=language,
            output_format=output_format,
            engine=engine,
        )
        progress(0.92, desc="Preparing result")
        comparison = _comparison_pair(str(image_path), enhanced)
        progress(1.0, desc="Done")
        return comparison, summary, enhanced, text_file
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def localize(language: str):
    t = TEXT[language]
    direction_class = ["ltr"] if language == "en" else ["rtl"]
    return (
        gr.HTML(value=_header(language)),
        gr.HTML(value=_hero(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Radio(label=t["engine"]),
        gr.Radio(label=t["format"]),
        gr.Slider(label=t["scale"]),
        gr.Button(value=t["process"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Textbox(label=t["text"], elem_classes=direction_class),
        gr.File(label=t["download"]),
        gr.File(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
html,body{height:100%;margin:0;overflow:hidden!important;background:#08111f}
body,.gradio-container{font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif!important}
.gradio-container{max-width:1480px!important;width:100%!important;height:100dvh!important;margin:0 auto!important;padding:8px 16px 10px!important;overflow:hidden!important;box-sizing:border-box!important}
footer,.footer,.built-with{display:none!important}
#header{height:52px!important;margin:0!important;overflow:hidden!important}
.topbar{height:52px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand-wrap{display:flex;align-items:center;gap:10px}.logo{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;color:#fff;font-weight:800;font-size:20px;background:linear-gradient(135deg,#0ea5e9,#14b8a6);box-shadow:0 8px 24px rgba(14,165,233,.22)}
.brand-name{font-size:1rem;font-weight:800}.brand-tag{font-size:.66rem;opacity:.52;margin-top:1px}.ready{display:flex;align-items:center;gap:7px;border:1px solid rgba(34,197,94,.22);background:rgba(34,197,94,.07);padding:6px 10px;border-radius:999px;font-size:.67rem}.ready span{width:7px;height:7px;border-radius:50%;background:#22c55e}
#hero{height:66px!important;overflow:hidden!important}.hero{text-align:center;padding:3px 0}.hero h1{font-size:1.55rem;line-height:1.4;margin:0;font-weight:800;letter-spacing:-.025em;background:linear-gradient(90deg,#38bdf8,#2dd4bf);-webkit-background-clip:text;color:transparent}.hero p{font-size:.73rem;opacity:.58;margin:3px 0 0}
#utility{height:36px!important;min-height:36px!important;margin:0 0 6px!important;gap:8px!important;align-items:center!important}
#workspace{height:calc(100dvh - 172px)!important;min-height:0!important;gap:14px!important;align-items:stretch!important;overflow:hidden!important}
.panel{height:100%!important;min-height:0!important;border:1px solid rgba(148,163,184,.14)!important;background:#0f1a2c!important;border-radius:20px!important;padding:12px!important;box-sizing:border-box!important;overflow:hidden!important}
#result-panel{display:flex!important;flex-direction:column!important;gap:8px!important}
#compare{flex:1 1 auto!important;min-height:0!important;height:auto!important;border-radius:14px!important;overflow:hidden!important}#compare>div{height:100%!important;min-height:0!important}
#result-info{flex:0 0 27%!important;min-height:0!important;gap:8px!important;margin:0!important}
#ocr-text{height:100%!important;min-height:0!important}#ocr-text textarea{height:calc(100% - 30px)!important;min-height:94px!important;resize:none!important}
#downloads{height:100%!important;min-height:0!important;display:flex!important;flex-direction:column!important;gap:6px!important}#downloads>div{flex:1!important;min-height:0!important;overflow:hidden!important}
#control-panel{display:flex!important;flex-direction:column!important;gap:7px!important}
#input-image{flex:1 1 auto!important;min-height:190px!important;max-height:300px!important;border:1px dashed rgba(45,212,191,.44)!important;border-radius:15px!important;overflow:hidden!important}#input-image>div{height:100%!important;min-height:0!important}#input-image img{object-fit:contain!important}
#sample-button button{height:34px!important;min-height:34px!important;border-radius:10px!important;font-size:.72rem!important}
#profile,#engine,#format,#scale{margin:0!important}
#process-button button{height:48px!important;min-height:48px!important;border-radius:12px!important;font-weight:800!important;background:linear-gradient(90deg,#0ea5e9,#14b8a6)!important;border:none!important}
.rtl,.rtl textarea,.rtl input{direction:rtl!important;text-align:right!important}.ltr,.ltr textarea,.ltr input{direction:ltr!important;text-align:left!important;font-family:'Inter',sans-serif!important}
@media(max-width:980px){html,body{overflow:auto!important}.gradio-container{height:auto!important;min-height:100dvh!important;overflow:visible!important;padding:10px!important}#workspace{height:auto!important;flex-wrap:wrap!important}.panel{height:auto!important;overflow:visible!important}#compare{height:380px!important}#result-info{min-height:250px!important}}
"""

THEME_JS = """
() => {
  const light = document.body.dataset.theme !== 'light';
  document.body.dataset.theme = light ? 'light' : 'dark';
  document.body.style.background = light ? '#eef4f8' : '#08111f';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(_header("fa"), elem_id="header")
    hero = gr.HTML(_hero("fa"), elem_id="hero")

    with gr.Row(elem_id="utility"):
        language = gr.Dropdown(
            choices=[("فارسی", "fa"), ("English", "en")],
            value="fa",
            label=None,
            show_label=False,
            scale=4,
        )
        theme = gr.Button("☼", scale=1)

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=8, elem_id="result-panel", elem_classes=["panel"]):
            comparison = gr.ImageSlider(
                label=TEXT["fa"]["compare"],
                type="filepath",
                interactive=False,
                elem_id="compare",
            )
            with gr.Row(elem_id="result-info"):
                output_text = gr.Textbox(
                    lines=5,
                    label=TEXT["fa"]["text"],
                    elem_id="ocr-text",
                    elem_classes=["rtl"],
                )
                with gr.Column(elem_id="downloads"):
                    enhanced_file = gr.File(label=TEXT["fa"]["download"])
                    text_file = gr.File(label=TEXT["fa"]["download_text"])

        with gr.Column(scale=5, elem_id="control-panel", elem_classes=["panel"]):
            input_image = gr.Image(
                type="filepath",
                sources=["upload", "clipboard"],
                label=TEXT["fa"]["upload"],
                elem_id="input-image",
            )
            sample_button = gr.Button(TEXT["fa"]["sample"], elem_id="sample-button")
            profile = gr.Radio(
                PROFILE_CHOICES["fa"],
                value="سند",
                label=TEXT["fa"]["profile"],
                elem_id="profile",
            )
            engine = gr.Radio(
                ENGINE_CHOICES,
                value="Text-Safe Pro",
                label=TEXT["fa"]["engine"],
                elem_id="engine",
            )
            output_format = gr.Radio(
                ["PNG", "JPG", "WEBP"],
                value="PNG",
                label=TEXT["fa"]["format"],
                elem_id="format",
            )
            scale = gr.Slider(
                1.0,
                3.0,
                value=2.0,
                step=0.5,
                label=TEXT["fa"]["scale"],
                elem_id="scale",
            )
            process_button = gr.Button(
                TEXT["fa"]["process"],
                variant="primary",
                elem_id="process-button",
            )

    sample_button.click(fn=lambda: SAMPLE_SCAN, outputs=[input_image])
    process_button.click(
        fn=run_single,
        inputs=[
            input_image,
            profile,
            engine,
            scale,
            output_format,
            language,
        ],
        outputs=[comparison, output_text, enhanced_file, text_file],
    )
    language.change(
        fn=localize,
        inputs=[language],
        outputs=[
            header,
            hero,
            input_image,
            sample_button,
            profile,
            engine,
            output_format,
            scale,
            process_button,
            comparison,
            output_text,
            enhanced_file,
            text_file,
        ],
    )
    theme.click(fn=None, js=THEME_JS)

if __name__ == "__main__":
    print("[APP] starting V9 on http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
