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

TEXT = {
    "fa": {
        "brand": "دقیق‌خوان",
        "tagline": "بهبود تصویر و OCR فارسی",
        "headline": "افزایش کیفیت تصویر فارسی",
        "sub": "تصویر را بارگذاری کنید، کیفیت را بالا ببرید و متن فارسی را استخراج کنید.",
        "upload": "تصویر را اینجا رها کنید یا برای انتخاب کلیک کنید",
        "sample": "استفاده از تصویر نمونه",
        "profile": "نوع تصویر",
        "format": "فرمت خروجی",
        "action": "تبدیل و بهبود تصویر",
        "compare": "قبل / بعد",
        "text": "متن استخراج‌شده",
        "download_image": "دریافت تصویر",
        "download_text": "دریافت متن",
        "ready": "موتور فارسی آماده",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian image enhancement & OCR",
        "headline": "Persian Image Upscaler",
        "sub": "Upload an image, enhance it, and extract Persian text.",
        "upload": "Drop an image here or click to browse",
        "sample": "Use sample image",
        "profile": "Image type",
        "format": "Output format",
        "action": "Enhance & Convert",
        "compare": "Before / After",
        "text": "Extracted text",
        "download_image": "Download image",
        "download_text": "Download text",
        "ready": "Persian engine ready",
    },
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_right(draw, x: int, y: int, text: str, font, fill: int = 32) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((max(30, x - width), y), text, font=font, fill=fill)


def _make_scan_sample() -> str:
    workdir = Path(tempfile.mkdtemp(prefix="daqiqkhan-sample-"))
    path = workdir / "persian-scan-sample.png"
    canvas = Image.new("L", (1100, 700), 244)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(34)
    body_font = _font(25)
    lines = [
        "نمونه سند فارسی برای ارزیابی OCR",
        "شماره سند: ۱۴۰۵-۰۶-۱۸",
        "نام کالا: گواهی سپرده کالایی",
        "مقدار: ۲۷٬۰۰۰٬۰۰۰ ریال",
        "این تصویر عمداً شبیه اسکن کم‌کیفیت ساخته شده است.",
    ]
    y = 85
    for index, line in enumerate(lines):
        _draw_right(draw, 1020, y, line, title_font if index == 0 else body_font)
        y += 96 if index == 0 else 78
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.55))
    canvas = canvas.rotate(0.3, resample=Image.Resampling.BICUBIC, fillcolor=246)
    canvas.save(path, format="PNG")
    return str(path)


SAMPLE_SCAN = _make_scan_sample()


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


def _header(language: str) -> str:
    t = TEXT[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='header' dir='{direction}'>"
        "<div class='brand'><span class='logo'>د</span><div>"
        f"<strong>{t['brand']}</strong><small>{t['tagline']}</small></div></div>"
        f"<div class='status'><i></i>{t['ready']}</div></div>"
    )


def _hero(language: str) -> str:
    t = TEXT[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='hero' dir='{direction}'>"
        f"<h1>{t['headline']}</h1>"
        f"<p>{t['sub']}</p>"
        "</div>"
    )


def run_single(image_path, profile, output_format, language, progress=gr.Progress()):
    if not image_path:
        raise gr.Error(
            "Please upload an image." if language == "en" else "لطفاً یک تصویر بارگذاری کنید."
        )
    progress(0.08, desc="Preparing")
    try:
        enhanced, _ocr_preview, summary, text_file = process_image(
            str(image_path),
            profile=profile,
            scale=2.0,
            language=language,
            output_format=output_format,
            engine="Text-Safe Pro",
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
    text_class = ["ltr"] if language == "en" else ["rtl"]
    return (
        gr.HTML(value=_header(language)),
        gr.HTML(value=_hero(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Radio(label=t["format"]),
        gr.Button(value=t["action"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Textbox(label=t["text"], elem_classes=text_class),
        gr.File(label=t["download_image"]),
        gr.File(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
html,body{height:100%;margin:0;overflow:hidden!important;background:#09111f}
body,.gradio-container{font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif!important}
.gradio-container{max-width:1420px!important;width:100%!important;height:100dvh!important;margin:0 auto!important;padding:8px 16px 12px!important;overflow:hidden!important;box-sizing:border-box!important}
footer,.footer,.built-with{display:none!important}
#header{height:48px!important;overflow:hidden!important;margin:0!important}.header{height:48px;display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:10px}.brand strong{display:block;font-size:1rem}.brand small{display:block;font-size:.63rem;opacity:.5}.logo{width:36px;height:36px;display:grid;place-items:center;border-radius:10px;background:linear-gradient(135deg,#7c3aed,#a855f7);color:#fff;font-weight:800;font-size:18px}.status{display:flex;align-items:center;gap:6px;border:1px solid rgba(34,197,94,.22);background:rgba(34,197,94,.07);padding:5px 9px;border-radius:999px;font-size:.66rem}.status i{width:6px;height:6px;border-radius:50%;background:#22c55e}
#hero{height:84px!important;overflow:hidden!important}.hero{text-align:center;padding:5px 0 0}.hero h1{font-size:1.55rem;line-height:1.5;margin:0;font-weight:800;color:#c084fc}.hero p{font-size:.73rem;opacity:.56;margin:3px 0 0}
#utility{height:34px!important;min-height:34px!important;margin:0 0 7px!important;gap:8px!important;align-items:center!important}
#workspace{height:calc(100dvh - 193px)!important;min-height:0!important;gap:22px!important;align-items:stretch!important;overflow:hidden!important}
.panel{height:100%!important;min-height:0!important;box-sizing:border-box!important;overflow:hidden!important}
#result{display:flex!important;flex-direction:column!important;gap:9px!important;background:transparent!important;border:none!important;padding:0!important}
#compare{flex:1 1 auto!important;min-height:0!important;height:auto!important;border-radius:18px!important;overflow:hidden!important;border:1px solid rgba(148,163,184,.14)!important;background:#111a2b!important;box-shadow:0 20px 55px rgba(0,0,0,.18)!important}#compare>div{height:100%!important;min-height:0!important}
#result-bottom{flex:0 0 23%!important;min-height:0!important;gap:8px!important;margin:0!important}
#text-box{height:100%!important;min-height:0!important;border-radius:14px!important}#text-box textarea{height:calc(100% - 30px)!important;min-height:80px!important;resize:none!important}
#files{height:100%!important;min-height:0!important;display:flex!important;flex-direction:column!important;gap:6px!important}#files>div{flex:1!important;min-height:0!important;overflow:hidden!important}
#controls{display:flex!important;flex-direction:column!important;gap:8px!important;border:1px solid rgba(139,92,246,.22)!important;background:#101a2d!important;border-radius:22px!important;padding:14px!important;box-shadow:0 24px 60px rgba(0,0,0,.22)!important}
#upload-title{height:44px!important;min-height:44px!important;margin:0!important;display:flex!important;align-items:center!important;justify-content:center!important;font-size:.9rem!important;font-weight:800!important;color:#e9d5ff!important}
#input{flex:1 1 auto!important;min-height:235px!important;border:1.5px dashed rgba(168,85,247,.48)!important;border-radius:17px!important;overflow:hidden!important;background:#0c1526!important}#input>div{height:100%!important;min-height:0!important}#input img{object-fit:contain!important}
#sample button{height:34px!important;min-height:34px!important;border-radius:10px!important;font-size:.72rem!important}
#profile,#format{margin:0!important}
#action button{height:58px!important;min-height:58px!important;border-radius:14px!important;font-size:1rem!important;font-weight:800!important;background:linear-gradient(90deg,#7c3aed,#a855f7)!important;border:none!important;box-shadow:0 10px 30px rgba(124,58,237,.25)!important}
#trust{height:24px!important;min-height:24px!important;margin:0!important;text-align:center!important;font-size:.64rem!important;opacity:.52!important}
.rtl,.rtl textarea,.rtl input{direction:rtl!important;text-align:right!important}.ltr,.ltr textarea,.ltr input{direction:ltr!important;text-align:left!important;font-family:'Inter',sans-serif!important}
@media(max-width:980px){html,body{overflow:auto!important}.gradio-container{height:auto!important;min-height:100dvh!important;overflow:visible!important;padding:10px!important}#workspace{height:auto!important;flex-wrap:wrap!important}.panel{height:auto!important;overflow:visible!important}#compare{height:360px!important}#result-bottom{min-height:220px!important}}
"""

THEME_JS = """
() => {
  const light = document.body.dataset.theme !== 'light';
  document.body.dataset.theme = light ? 'light' : 'dark';
  document.body.style.background = light ? '#f5f3ff' : '#09111f';
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
        with gr.Column(scale=8, elem_id="result", elem_classes=["panel"]):
            comparison = gr.ImageSlider(
                label=TEXT["fa"]["compare"],
                type="filepath",
                interactive=False,
                elem_id="compare",
            )
            with gr.Row(elem_id="result-bottom"):
                output_text = gr.Textbox(
                    lines=4,
                    label=TEXT["fa"]["text"],
                    elem_id="text-box",
                    elem_classes=["rtl"],
                )
                with gr.Column(elem_id="files"):
                    enhanced_file = gr.File(label=TEXT["fa"]["download_image"])
                    text_file = gr.File(label=TEXT["fa"]["download_text"])

        with gr.Column(scale=5, elem_id="controls", elem_classes=["panel"]):
            gr.HTML("<div id='upload-title'>تصویر فارسی را بارگذاری کنید</div>")
            input_image = gr.Image(
                type="filepath",
                sources=["upload", "clipboard"],
                label=TEXT["fa"]["upload"],
                elem_id="input",
            )
            sample_button = gr.Button(TEXT["fa"]["sample"], elem_id="sample")
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
            process_button = gr.Button(
                TEXT["fa"]["action"],
                variant="primary",
                elem_id="action",
            )
            gr.HTML("<div id='trust'>پردازش امن · مخصوص متن فارسی · خروجی بدون واترمارک</div>")

    sample_button.click(fn=lambda: SAMPLE_SCAN, outputs=[input_image])
    process_button.click(
        fn=run_single,
        inputs=[input_image, profile, output_format, language],
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
            output_format,
            process_button,
            comparison,
            output_text,
            enhanced_file,
            text_file,
        ],
    )
    theme.click(fn=None, js=THEME_JS)

if __name__ == "__main__":
    print("[APP] starting V9 reference-inspired workspace on http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
