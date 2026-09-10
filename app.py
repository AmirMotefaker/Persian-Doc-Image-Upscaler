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
        "headline": "تصویر فارسی را واضح کن؛ متن را دقیق بگیر",
        "upload": "تصویر را اینجا رها کنید یا کلیک کنید",
        "sample": "نمونه آماده",
        "profile": "نوع تصویر",
        "action": "تبدیل و بهبود فایل",
        "compare": "قبل / بعد",
        "text": "متن استخراج‌شده",
        "download_image": "دریافت تصویر",
        "download_text": "دریافت متن",
        "ready": "آماده",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian image enhancement & OCR",
        "headline": "Enhance Persian images and extract text",
        "upload": "Drop an image here or click to browse",
        "sample": "Try sample",
        "profile": "Image type",
        "action": "Enhance & Convert",
        "compare": "Before / After",
        "text": "Extracted text",
        "download_image": "Download image",
        "download_text": "Download text",
        "ready": "Ready",
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


def _headline(language: str) -> str:
    direction = "ltr" if language == "en" else "rtl"
    return f"<div class='headline' dir='{direction}'>{TEXT[language]['headline']}</div>"


def run_single(image_path, profile, language, progress=gr.Progress()):
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
            output_format="PNG",
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
        gr.HTML(value=_headline(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Button(value=t["action"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Textbox(label=t["text"], elem_classes=text_class),
        gr.File(label=t["download_image"]),
        gr.File(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
html,body{height:100%;margin:0;overflow:hidden!important;background:#081019}
body,.gradio-container{font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif!important}
.gradio-container{max-width:1480px!important;width:100%!important;height:100dvh!important;margin:0 auto!important;padding:8px 14px!important;overflow:hidden!important;box-sizing:border-box!important}
footer,.footer,.built-with{display:none!important}
#header{height:46px!important;overflow:hidden!important;margin:0!important}.header{height:46px;display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:9px}.brand strong{display:block;font-size:1rem}.brand small{display:block;font-size:.62rem;opacity:.5}.logo{width:34px;height:34px;display:grid;place-items:center;border-radius:10px;background:linear-gradient(135deg,#06b6d4,#14b8a6);color:#fff;font-weight:800;font-size:18px}.status{display:flex;align-items:center;gap:6px;border:1px solid rgba(34,197,94,.22);background:rgba(34,197,94,.07);padding:5px 9px;border-radius:999px;font-size:.66rem}.status i{width:6px;height:6px;border-radius:50%;background:#22c55e}
#topbar{height:36px!important;min-height:36px!important;margin:0 0 5px!important;gap:8px!important;align-items:center!important}.headline{text-align:center;font-size:1.05rem;font-weight:800;color:#67e8f9;line-height:34px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#workspace{height:calc(100dvh - 103px)!important;min-height:0!important;gap:12px!important;align-items:stretch!important;overflow:hidden!important}.panel{height:100%!important;min-height:0!important;border:1px solid rgba(148,163,184,.14)!important;background:#0e1725!important;border-radius:18px!important;padding:10px!important;box-sizing:border-box!important;overflow:hidden!important}
#result{display:flex!important;flex-direction:column!important;gap:8px!important}#compare{flex:1 1 auto!important;min-height:0!important;height:auto!important;border-radius:13px!important;overflow:hidden!important}#compare>div{height:100%!important;min-height:0!important}#bottom{flex:0 0 29%!important;min-height:0!important;gap:8px!important;margin:0!important}#text-box{height:100%!important;min-height:0!important}#text-box textarea{height:calc(100% - 30px)!important;min-height:90px!important;resize:none!important}#files{height:100%!important;min-height:0!important;display:flex!important;flex-direction:column!important;gap:6px!important}#files>div{flex:1!important;min-height:0!important;overflow:hidden!important}
#controls{display:flex!important;flex-direction:column!important;gap:8px!important}#input{flex:1 1 auto!important;min-height:260px!important;border:1px dashed rgba(34,211,238,.42)!important;border-radius:14px!important;overflow:hidden!important}#input>div{height:100%!important;min-height:0!important}#input img{object-fit:contain!important}#sample button{height:34px!important;min-height:34px!important;border-radius:9px!important;font-size:.72rem!important}#profile{margin:0!important}#action button{height:56px!important;min-height:56px!important;border-radius:13px!important;font-size:1rem!important;font-weight:800!important;background:linear-gradient(90deg,#06b6d4,#14b8a6)!important;border:none!important;box-shadow:0 8px 28px rgba(6,182,212,.18)!important}.rtl,.rtl textarea,.rtl input{direction:rtl!important;text-align:right!important}.ltr,.ltr textarea,.ltr input{direction:ltr!important;text-align:left!important;font-family:'Inter',sans-serif!important}
@media(max-width:980px){html,body{overflow:auto!important}.gradio-container{height:auto!important;min-height:100dvh!important;overflow:visible!important;padding:10px!important}#workspace{height:auto!important;flex-wrap:wrap!important}.panel{height:auto!important;overflow:visible!important}#compare{height:360px!important}#bottom{min-height:240px!important}}
"""

THEME_JS = """
() => {
  const light = document.body.dataset.theme !== 'light';
  document.body.dataset.theme = light ? 'light' : 'dark';
  document.body.style.background = light ? '#eef4f6' : '#081019';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(_header("fa"), elem_id="header")
    with gr.Row(elem_id="topbar"):
        headline = gr.HTML(_headline("fa"), scale=7)
        language = gr.Dropdown(
            choices=[("فارسی", "fa"), ("English", "en")],
            value="fa",
            label=None,
            show_label=False,
            scale=2,
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
            with gr.Row(elem_id="bottom"):
                output_text = gr.Textbox(
                    lines=5,
                    label=TEXT["fa"]["text"],
                    elem_id="text-box",
                    elem_classes=["rtl"],
                )
                with gr.Column(elem_id="files"):
                    enhanced_file = gr.File(label=TEXT["fa"]["download_image"])
                    text_file = gr.File(label=TEXT["fa"]["download_text"])

        with gr.Column(scale=4, elem_id="controls", elem_classes=["panel"]):
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
            process_button = gr.Button(
                TEXT["fa"]["action"],
                variant="primary",
                elem_id="action",
            )

    sample_button.click(fn=lambda: SAMPLE_SCAN, outputs=[input_image])
    process_button.click(
        fn=run_single,
        inputs=[input_image, profile, language],
        outputs=[comparison, output_text, enhanced_file, text_file],
    )
    language.change(
        fn=localize,
        inputs=[language],
        outputs=[
            header,
            headline,
            input_image,
            sample_button,
            profile,
            process_button,
            comparison,
            output_text,
            enhanced_file,
            text_file,
        ],
    )
    theme.click(fn=None, js=THEME_JS)

if __name__ == "__main__":
    print("[APP] starting V9 minimal workspace on http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
