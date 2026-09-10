from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from persian_upscaler.service import process_image

PROFILE_CHOICES = {
    "fa": [("سند", "سند"), ("تصویر واضح", "طبیعی"), ("اسکن ضعیف", "اسکن ضعیف")],
    "en": [("Document", "سند"), ("Clean image", "طبیعی"), ("Weak scan", "اسکن ضعیف")],
}

TEXT = {
    "fa": {
        "brand": "دقیق‌خوان",
        "tagline": "بهبود کیفیت تصویر و OCR تخصصی فارسی",
        "ready": "موتور فارسی آماده",
        "upload": "تصویر را اینجا رها کنید یا کلیک کنید",
        "sample": "استفاده از نمونه اسکن",
        "profile": "نوع تصویر",
        "format": "فرمت خروجی",
        "scale": "افزایش ابعاد",
        "process": "بهبود تصویر و استخراج متن",
        "compare": "قبل / بعد",
        "ocr": "نمای OCR",
        "text": "متن استخراج‌شده",
        "download": "دریافت تصویر",
        "download_text": "دریافت متن",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian Image Enhancement & OCR",
        "ready": "Persian engine ready",
        "upload": "Drop an image here or click to browse",
        "sample": "Use scan sample",
        "profile": "Image type",
        "format": "Output format",
        "scale": "Upscale factor",
        "process": "Enhance image & extract text",
        "compare": "Before / After",
        "ocr": "OCR view",
        "text": "Extracted text",
        "download": "Download image",
        "download_text": "Download text",
    },
}


def _brand(language: str) -> str:
    t = TEXT[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='brandbar' dir='{direction}'>"
        "<div class='brand-left'><div class='logo'>د</div><div>"
        f"<div class='brand-name'>{t['brand']}</div>"
        f"<div class='brand-tag'>{t['tagline']}</div></div></div>"
        f"<div class='ready'><i></i>{t['ready']}</div></div>"
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


def run_single(image_path, profile, scale, output_format, language):
    if not image_path:
        raise gr.Error(
            "Please upload an image containing Persian text."
            if language == "en"
            else "لطفاً یک تصویر دارای متن فارسی بارگذاری کنید."
        )
    try:
        enhanced, ocr_preview, summary, text_file = process_image(
            str(image_path),
            profile=profile,
            scale=float(scale),
            language=language,
            output_format=output_format,
        )
        return (
            _comparison_pair(str(image_path), enhanced),
            ocr_preview,
            summary,
            enhanced,
            text_file,
        )
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def localize(language: str):
    t = TEXT[language]
    text_class = ["ltr"] if language == "en" else ["rtl"]
    return (
        gr.HTML(value=_brand(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Radio(label=t["format"]),
        gr.Slider(label=t["scale"]),
        gr.Button(value=t["process"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Image(label=t["ocr"]),
        gr.Textbox(label=t["text"], elem_classes=text_class),
        gr.File(label=t["download"]),
        gr.File(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
html, body {height:100%; margin:0; overflow:hidden !important; background:#0b1020;}
body, .gradio-container {font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif !important;}
.gradio-container {max-width:1460px !important; width:100% !important; height:100dvh !important; margin:0 auto !important; padding:10px 16px !important; overflow:hidden !important; box-sizing:border-box !important;}
footer, .footer, .built-with {display:none !important;}
#brand {height:52px !important; margin:0 0 8px !important; overflow:hidden !important;}
.brandbar {height:52px; display:flex; align-items:center; justify-content:space-between; gap:14px;}
.brand-left {display:flex; align-items:center; gap:10px;}
.logo {width:38px; height:38px; border-radius:12px; display:grid; place-items:center; color:#fff; font-size:20px; font-weight:800; background:linear-gradient(135deg,#2563eb,#7c3aed); box-shadow:0 8px 24px rgba(37,99,235,.23);}
.brand-name {font-size:1.04rem; font-weight:800; line-height:1.15;}
.brand-tag {font-size:.68rem; opacity:.52; margin-top:2px;}
.ready {display:flex; align-items:center; gap:7px; border:1px solid rgba(34,197,94,.22); background:rgba(34,197,94,.07); padding:6px 10px; border-radius:999px; font-size:.68rem;}
.ready i {width:7px; height:7px; border-radius:50%; background:#22c55e;}
#top-controls {height:40px !important; min-height:40px !important; margin:0 0 8px !important; gap:8px !important; align-items:center !important;}
#workspace {height:calc(100dvh - 128px) !important; min-height:0 !important; gap:12px !important; align-items:stretch !important; overflow:hidden !important;}
.card {height:100% !important; min-height:0 !important; overflow:hidden !important; border:1px solid rgba(148,163,184,.14) !important; background:#101827 !important; border-radius:18px !important; padding:12px !important; box-sizing:border-box !important;}
#result-card {display:flex !important; flex-direction:column !important; gap:8px !important;}
#compare {flex:1 1 64% !important; min-height:0 !important; height:auto !important;}
#compare > div {height:100% !important; min-height:0 !important;}
#result-bottom {flex:0 0 28% !important; min-height:0 !important; gap:8px !important; margin:0 !important;}
#ocr-preview, #ocr-text {height:100% !important; min-height:0 !important;}
#ocr-preview > div {height:100% !important; min-height:0 !important;}
#ocr-text textarea {height:calc(100% - 30px) !important; min-height:92px !important; resize:none !important;}
#downloads {flex:0 0 44px !important; min-height:44px !important; margin:0 !important; gap:8px !important;}
#downloads > div {min-height:40px !important; max-height:44px !important; overflow:hidden !important;}
#control-card {display:flex !important; flex-direction:column !important; gap:8px !important;}
#input-image {flex:1 1 auto !important; min-height:210px !important; max-height:330px !important; border:1px dashed rgba(96,165,250,.42) !important; border-radius:14px !important; overflow:hidden !important;}
#input-image > div {height:100% !important; min-height:0 !important;}
#input-image img {object-fit:contain !important;}
#sample-button button {height:36px !important; min-height:36px !important; border-radius:10px !important; font-size:.74rem !important;}
#profile, #format, #scale {margin:0 !important;}
#process-button button {height:46px !important; min-height:46px !important; border-radius:12px !important; font-weight:800 !important; background:linear-gradient(90deg,#2563eb,#7c3aed) !important; border:none !important;}
.rtl, .rtl textarea, .rtl input {direction:rtl !important; text-align:right !important;}
.ltr, .ltr textarea, .ltr input {direction:ltr !important; text-align:left !important; font-family:'Inter',sans-serif !important;}
@media(max-width:980px){html,body{overflow:auto !important}.gradio-container{height:auto !important;min-height:100dvh !important;overflow:visible !important;padding:10px !important}#workspace{height:auto !important;flex-wrap:wrap !important}.card{height:auto !important;overflow:visible !important}#compare{height:380px !important}#result-bottom{min-height:260px !important}}
"""

THEME_JS = """
() => {
  const light = document.body.dataset.theme !== 'light';
  document.body.dataset.theme = light ? 'light' : 'dark';
  document.body.style.background = light ? '#f4f7fb' : '#0b1020';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    brand = gr.HTML(_brand("fa"), elem_id="brand")

    with gr.Row(elem_id="top-controls"):
        language = gr.Radio(
            [("فارسی", "fa"), ("English", "en")],
            value="fa",
            label=None,
            scale=4,
        )
        theme = gr.Button("☼", scale=1)

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=8, elem_id="result-card", elem_classes=["card"]):
            comparison = gr.ImageSlider(
                label=TEXT["fa"]["compare"],
                type="filepath",
                interactive=False,
                elem_id="compare",
            )
            with gr.Row(elem_id="result-bottom"):
                ocr_preview = gr.Image(
                    type="filepath",
                    label=TEXT["fa"]["ocr"],
                    interactive=False,
                    elem_id="ocr-preview",
                )
                output_text = gr.Textbox(
                    lines=5,
                    label=TEXT["fa"]["text"],
                    elem_id="ocr-text",
                    elem_classes=["rtl"],
                )
            with gr.Row(elem_id="downloads"):
                enhanced_file = gr.File(label=TEXT["fa"]["download"])
                text_file = gr.File(label=TEXT["fa"]["download_text"])

        with gr.Column(scale=5, elem_id="control-card", elem_classes=["card"]):
            input_image = gr.Image(
                type="filepath",
                sources=["upload", "clipboard"],
                label=TEXT["fa"]["upload"],
                elem_id="input-image",
            )
            sample_button = gr.Button(
                TEXT["fa"]["sample"],
                elem_id="sample-button",
            )
            profile = gr.Radio(
                PROFILE_CHOICES["fa"],
                value="سند",
                label=TEXT["fa"]["profile"],
                elem_id="profile",
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
        inputs=[input_image, profile, scale, output_format, language],
        outputs=[comparison, ocr_preview, output_text, enhanced_file, text_file],
    )
    language.change(
        fn=localize,
        inputs=[language],
        outputs=[
            brand,
            input_image,
            sample_button,
            profile,
            output_format,
            scale,
            process_button,
            comparison,
            ocr_preview,
            output_text,
            enhanced_file,
            text_file,
        ],
    )
    theme.click(fn=None, js=THEME_JS)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        css=CSS,
        prevent_thread_lock=True,
    )
    print("[APP] server started; press Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[APP] shutting down", flush=True)
        demo.close()
