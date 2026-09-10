from __future__ import annotations

import html
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
        "tagline": "افزایش کیفیت تصویر و OCR دقیق فارسی",
        "headline": "تصویر فارسی را واضح‌تر کن",
        "sub": "یک تصویر بارگذاری کن؛ خروجی HD و متن فارسی ساختاریافته تحویل بگیر.",
        "upload": "تصویر را بکش و اینجا رها کن یا کلیک کن",
        "sample": "نمونه اسکن",
        "profile": "نوع تصویر",
        "action": "افزایش کیفیت و استخراج متن",
        "compare": "قبل / بعد",
        "text": "متن استخراج‌شده",
        "download_image": "دریافت تصویر HD",
        "download_text": "دریافت متن",
        "ready": "آماده",
        "empty": "بعد از پردازش، متن فارسی اینجا نمایش داده می‌شود.",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian image enhancement & OCR",
        "headline": "Make Persian images clearer",
        "sub": "Upload once; get an HD image and structured Persian text.",
        "upload": "Drag & drop an image or click to browse",
        "sample": "Scan sample",
        "profile": "Image type",
        "action": "Enhance image & extract text",
        "compare": "Before / After",
        "text": "Extracted text",
        "download_image": "Download HD image",
        "download_text": "Download text",
        "ready": "Ready",
        "empty": "Extracted Persian text will appear here.",
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
            original = original.resize(target_size, Image.Resampling.BICUBIC)
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
        f"<h1>{t['headline']}</h1><p>{t['sub']}</p>"
        "</div>"
    )


def _render_text(text: str, language: str) -> str:
    direction = "ltr" if language == "en" else "rtl"
    if not text.strip():
        text = TEXT[language]["empty"]
    escaped = html.escape(text)
    return (
        f"<div class='ocr-output' dir='{direction}'>"
        f"<pre>{escaped}</pre></div>"
    )


def run_single(image_path, profile, language, progress=gr.Progress()):
    if not image_path:
        raise gr.Error(
            "Please upload an image." if language == "en" else "لطفاً یک تصویر بارگذاری کنید."
        )

    progress(0.05, desc="Preparing")
    try:
        enhanced, _ocr_preview, canonical_text, text_file = process_image(
            str(image_path),
            profile=profile,
            scale=4.0,
            language=language,
            output_format="PNG",
            engine="Super-Resolution Pro",
        )
        progress(0.92, desc="Preparing result")
        comparison = _comparison_pair(str(image_path), enhanced)
        progress(1.0, desc="Done")
        return (
            comparison,
            _render_text(canonical_text, language),
            enhanced,
            text_file,
        )
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def localize(language: str):
    t = TEXT[language]
    return (
        gr.HTML(value=_header(language)),
        gr.HTML(value=_hero(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Button(value=t["action"]),
        gr.ImageSlider(label=t["compare"]),
        gr.HTML(value=_render_text("", language)),
        gr.DownloadButton(label=t["download_image"]),
        gr.DownloadButton(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
:root{--bg:#08111e;--panel:#0d1a2b;--panel2:#101f34;--border:rgba(148,163,184,.14);--muted:#94a3b8;--text:#f8fafc;--accent:#14b8a6;--accent2:#22d3ee}
body[data-theme='light']{--bg:#f5f7fb;--panel:#fff;--panel2:#f8fafc;--border:rgba(15,23,42,.12);--muted:#64748b;--text:#0f172a;--accent:#0f766e;--accent2:#0891b2}
html,body{min-height:100%;margin:0;overflow-x:hidden!important;overflow-y:auto!important;background:var(--bg)!important;color:var(--text)!important}
body,.gradio-container{font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif!important}
.gradio-container{max-width:1480px!important;width:100%!important;min-height:100dvh!important;height:auto!important;margin:0 auto!important;padding:8px 18px 18px!important;overflow:visible!important;box-sizing:border-box!important;background:var(--bg)!important}
footer,.footer,.built-with{display:none!important}
#header{height:48px!important;overflow:hidden!important;margin:0!important}.header{height:48px;display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:10px}.brand strong{display:block;font-size:1.02rem}.brand small{display:block;font-size:.63rem;color:var(--muted)}.logo{width:36px;height:36px;display:grid;place-items:center;border-radius:11px;background:linear-gradient(135deg,var(--accent2),var(--accent));color:#fff;font-weight:800;font-size:18px}.status{display:flex;align-items:center;gap:6px;border:1px solid rgba(34,197,94,.22);background:rgba(34,197,94,.07);padding:5px 9px;border-radius:999px;font-size:.66rem}.status i{width:6px;height:6px;border-radius:50%;background:#22c55e}
#top{height:70px!important;min-height:70px!important;margin:0 0 6px!important;align-items:center!important}.hero{text-align:center}.hero h1{font-size:1.65rem;line-height:1.35;margin:0;font-weight:800;background:linear-gradient(90deg,var(--accent2),var(--accent));-webkit-background-clip:text;color:transparent}.hero p{font-size:.75rem;color:var(--muted);margin:4px 0 0}.mini-controls{display:flex;gap:6px!important}
#workspace{min-height:calc(100dvh - 144px)!important;height:auto!important;gap:18px!important;align-items:stretch!important;overflow:visible!important}.panel{min-height:0!important;box-sizing:border-box!important;overflow:visible!important}
#result{display:flex!important;flex-direction:column!important;gap:9px!important}.result-shell{min-height:680px!important;border:1px solid var(--border)!important;background:var(--panel)!important;border-radius:20px!important;padding:10px!important;box-sizing:border-box!important;display:flex!important;flex-direction:column!important;gap:8px!important}
#compare{height:500px!important;min-height:420px!important;border-radius:15px!important;overflow:hidden!important;background:var(--panel2)!important}#compare>div{height:100%!important;min-height:0!important}
#result-bottom{min-height:170px!important;gap:8px!important;margin:0!important}.ocr-wrap{min-height:170px!important;border:1px solid var(--border)!important;background:var(--panel2)!important;border-radius:13px!important;padding:9px!important;overflow:auto!important}.ocr-output{height:100%;overflow:auto}.ocr-output pre{margin:0;white-space:pre-wrap;tab-size:8;font-family:'Vazirmatn','Inter',sans-serif;font-size:.78rem;line-height:1.75;color:var(--text)}#downloads{min-height:170px!important;display:flex!important;flex-direction:column!important;gap:7px!important}#downloads button{flex:1!important;min-height:40px!important;border-radius:11px!important}
#controls{display:flex!important;flex-direction:column!important;gap:9px!important;border:1px solid var(--border)!important;background:var(--panel)!important;border-radius:20px!important;padding:14px!important;min-height:680px!important}.upload-head{text-align:center;font-size:.94rem;font-weight:800;margin:1px 0 2px}.upload-sub{text-align:center;font-size:.67rem;color:var(--muted);margin-bottom:2px}
#input{height:410px!important;min-height:320px!important;border:1.5px dashed color-mix(in srgb,var(--accent2) 55%,transparent)!important;border-radius:16px!important;overflow:hidden!important;background:var(--panel2)!important}#input>div{height:100%!important;min-height:0!important}#input img{object-fit:contain!important}#sample button{height:34px!important;min-height:34px!important;border-radius:10px!important;font-size:.72rem!important}#profile{margin:0!important}#action button{height:60px!important;min-height:60px!important;border-radius:14px!important;font-size:1rem!important;font-weight:800!important;background:linear-gradient(90deg,var(--accent2),var(--accent))!important;border:none!important;box-shadow:0 10px 30px rgba(20,184,166,.2)!important}.trust{text-align:center;font-size:.64rem;color:var(--muted);margin-top:1px}
@media(min-width:1100px) and (min-height:900px){#workspace{min-height:calc(100dvh - 144px)!important}.result-shell,#controls{min-height:calc(100dvh - 162px)!important}#compare{height:calc(100dvh - 365px)!important}}
@media(max-width:980px){.gradio-container{padding:10px!important}#workspace{height:auto!important;min-height:auto!important;flex-wrap:wrap!important}.panel{height:auto!important}.result-shell,#controls{min-height:auto!important}#compare{height:360px!important}#result-bottom{min-height:230px!important}}
"""

THEME_JS = """
() => {
  const body = document.body;
  body.dataset.theme = body.dataset.theme === 'light' ? 'dark' : 'light';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(_header("fa"), elem_id="header")

    with gr.Row(elem_id="top"):
        hero = gr.HTML(_hero("fa"), scale=8)
        with gr.Row(scale=2, elem_classes=["mini-controls"]):
            language = gr.Dropdown(
                choices=[("فارسی", "fa"), ("English", "en")],
                value="fa",
                label=None,
                show_label=False,
            )
            theme = gr.Button("◐")

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=8, elem_id="result", elem_classes=["panel"]):
            with gr.Column(elem_classes=["result-shell"]):
                comparison = gr.ImageSlider(
                    label=TEXT["fa"]["compare"],
                    type="filepath",
                    interactive=False,
                    elem_id="compare",
                )
                with gr.Row(elem_id="result-bottom"):
                    with gr.Column(scale=7, elem_classes=["ocr-wrap"]):
                        text_html = gr.HTML(_render_text("", "fa"))
                    with gr.Column(scale=3, elem_id="downloads"):
                        image_download = gr.DownloadButton(
                            TEXT["fa"]["download_image"]
                        )
                        text_download = gr.DownloadButton(
                            TEXT["fa"]["download_text"]
                        )

        with gr.Column(scale=5, elem_id="controls", elem_classes=["panel"]):
            gr.HTML(
                "<div class='upload-head'>تصویر فارسی را بارگذاری کنید</div>"
                "<div class='upload-sub'>PNG · JPG · JPEG · WEBP · BMP · TIFF</div>"
            )
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
            gr.HTML(
                "<div class='trust'>پردازش محلی · OCR فارسی · خروجی HD غیرمولد</div>"
            )

    sample_button.click(fn=lambda: SAMPLE_SCAN, outputs=[input_image])
    process_button.click(
        fn=run_single,
        inputs=[input_image, profile, language],
        outputs=[comparison, text_html, image_download, text_download],
        show_progress="full",
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
            process_button,
            comparison,
            text_html,
            image_download,
            text_download,
        ],
    )
    theme.click(fn=None, js=THEME_JS)

if __name__ == "__main__":
    print("[APP] starting redesigned workspace on http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
