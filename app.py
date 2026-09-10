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
        "tagline": "افزایش کیفیت و OCR تخصصی فارسی",
        "headline": "تصویر فارسی را واضح و خوانا کن",
        "upload": "تصویر را بکش و اینجا رها کن یا کلیک کن",
        "sample": "استفاده از نمونه",
        "profile": "نوع تصویر",
        "action": "افزایش کیفیت و استخراج متن",
        "compare": "قبل / بعد",
        "ocr": "متن استخراج‌شده",
        "download_image": "دریافت تصویر HD",
        "download_text": "دریافت متن",
        "ready": "آماده",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian image enhancement & OCR",
        "headline": "Make Persian images crisp and readable",
        "upload": "Drag & drop an image or click to browse",
        "sample": "Use sample",
        "profile": "Image type",
        "action": "Enhance image & extract text",
        "compare": "Before / After",
        "ocr": "Extracted text",
        "download_image": "Download HD image",
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


def _draw_right(draw, x: int, y: int, text: str, font) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((max(24, x - (bbox[2] - bbox[0])), y), text, font=font, fill=32)


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
        "این تصویر شبیه اسکن کم‌کیفیت ساخته شده است.",
    ]
    y = 85
    for index, line in enumerate(lines):
        _draw_right(draw, 1020, y, line, title_font if index == 0 else body_font)
        y += 96 if index == 0 else 78
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.55))
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
        f"<div class='headline'>{t['headline']}</div>"
        f"<div class='status'><i></i>{t['ready']}</div></div>"
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
        progress(0.90, desc="Preparing result")
        comparison = _comparison_pair(str(image_path), enhanced)
        progress(1.0, desc="Done")
        return comparison, canonical_text, enhanced, text_file
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def localize(language: str):
    t = TEXT[language]
    rtl_class = ["rtl"] if language == "fa" else ["ltr"]
    return (
        gr.HTML(value=_header(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Button(value=t["action"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Textbox(label=t["ocr"], elem_classes=rtl_class),
        gr.DownloadButton(label=t["download_image"]),
        gr.DownloadButton(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
:root{--bg:#07111f;--panel:#0d1a2b;--panel2:#111f33;--border:rgba(148,163,184,.16);--muted:#94a3b8;--text:#f8fafc;--accent:#14b8a6;--accent2:#22d3ee}
body[data-theme='light']{--bg:#f6f8fb;--panel:#fff;--panel2:#f8fafc;--border:rgba(15,23,42,.12);--muted:#64748b;--text:#0f172a;--accent:#0f766e;--accent2:#0891b2}
*{box-sizing:border-box!important}html,body{height:100%;margin:0;overflow:hidden!important;background:var(--bg)!important;color:var(--text)!important}body,.gradio-container{font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif!important}.gradio-container{max-width:1500px!important;width:100%!important;height:100dvh!important;margin:0 auto!important;padding:8px 14px!important;overflow:hidden!important;background:var(--bg)!important}footer,.footer,.built-with{display:none!important}
#header{height:52px!important;min-height:52px!important;margin:0 0 6px!important}.header{height:52px;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:16px}.brand{display:flex;align-items:center;gap:9px}.brand strong{display:block;font-size:.98rem}.brand small{display:block;font-size:.58rem;color:var(--muted)}.logo{width:34px;height:34px;display:grid;place-items:center;border-radius:10px;background:linear-gradient(135deg,var(--accent2),var(--accent));color:#fff;font-weight:800}.headline{font-size:1.15rem;font-weight:800;white-space:nowrap;background:linear-gradient(90deg,var(--accent2),var(--accent));-webkit-background-clip:text;color:transparent}.status{justify-self:end;display:flex;align-items:center;gap:5px;border:1px solid rgba(34,197,94,.22);background:rgba(34,197,94,.07);padding:4px 8px;border-radius:999px;font-size:.61rem}.status i{width:6px;height:6px;border-radius:50%;background:#22c55e}
#workspace{height:calc(100dvh - 72px)!important;min-height:0!important;gap:12px!important;align-items:stretch!important;overflow:hidden!important}.panel{height:100%!important;min-height:0!important;overflow:hidden!important}
#result-shell{height:100%!important;min-height:0!important;display:grid!important;grid-template-rows:minmax(0,1fr) 150px!important;gap:8px!important;background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:18px!important;padding:9px!important}#compare{height:100%!important;min-height:0!important;border-radius:14px!important;overflow:hidden!important;background:var(--panel2)!important}#compare>div{height:100%!important;min-height:0!important}#result-bottom{height:150px!important;min-height:0!important;gap:8px!important;margin:0!important}#ocr{height:150px!important;min-height:0!important}#ocr textarea{height:108px!important;min-height:108px!important;resize:none!important;font-size:.75rem!important;line-height:1.65!important}.rtl textarea{direction:rtl!important;text-align:right!important}.ltr textarea{direction:ltr!important;text-align:left!important;font-family:'Inter',sans-serif!important}#downloads{height:150px!important;display:grid!important;grid-template-rows:1fr 1fr!important;gap:7px!important}#downloads button{height:100%!important;min-height:0!important;border-radius:11px!important;font-size:.76rem!important;font-weight:700!important}
#controls{height:100%!important;min-height:0!important;display:grid!important;grid-template-rows:auto minmax(0,1fr) 34px 70px 58px auto!important;gap:8px!important;background:var(--panel)!important;border:1px solid var(--border)!important;border-radius:18px!important;padding:11px!important;overflow:hidden!important}.upload-head{text-align:center;font-size:.90rem;font-weight:800}.upload-sub{text-align:center;font-size:.59rem;color:var(--muted)}#input{height:100%!important;min-height:0!important;border:1.5px dashed rgba(34,211,238,.42)!important;border-radius:14px!important;overflow:hidden!important;background:var(--panel2)!important}#input>div{height:100%!important;min-height:0!important}#input img{object-fit:contain!important}#sample button{height:34px!important;min-height:34px!important;border-radius:9px!important;font-size:.70rem!important}#profile{height:70px!important;min-height:70px!important;margin:0!important}#profile .wrap{gap:6px!important}#action button{height:58px!important;min-height:58px!important;border-radius:13px!important;font-size:.95rem!important;font-weight:800!important;background:linear-gradient(90deg,var(--accent2),var(--accent))!important;border:none!important;box-shadow:0 8px 24px rgba(20,184,166,.18)!important}.trust{text-align:center;font-size:.58rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.mini-controls{position:fixed!important;top:10px!important;left:50%!important;transform:translateX(-50%)!important;width:210px!important;z-index:20!important;gap:6px!important}.mini-controls button,.mini-controls input{min-height:32px!important;height:32px!important}
@media(max-height:820px){#header{height:44px!important;min-height:44px!important}.header{height:44px}.brand small{display:none}#workspace{height:calc(100dvh - 60px)!important}#result-shell{grid-template-rows:minmax(0,1fr) 126px!important}#result-bottom,#downloads,#ocr{height:126px!important}#ocr textarea{height:84px!important;min-height:84px!important}#controls{grid-template-rows:auto minmax(0,1fr) 30px 62px 52px auto!important}#action button{height:52px!important;min-height:52px!important}}
@media(max-width:1050px){.headline{font-size:.90rem}.brand small{display:none}.gradio-container{padding:6px 8px!important}#workspace{gap:8px!important}#controls{padding:8px!important}}
"""

THEME_JS = """
() => {
  const body = document.body;
  body.dataset.theme = body.dataset.theme === 'light' ? 'dark' : 'light';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(_header("fa"), elem_id="header")

    with gr.Row(elem_classes=["mini-controls"]):
        language = gr.Dropdown(
            choices=[("فارسی", "fa"), ("English", "en")],
            value="fa",
            label=None,
            show_label=False,
            scale=3,
        )
        theme = gr.Button("◐", scale=1)

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=8, elem_classes=["panel"]):
            with gr.Column(elem_id="result-shell"):
                comparison = gr.ImageSlider(
                    label=TEXT["fa"]["compare"],
                    type="filepath",
                    interactive=False,
                    elem_id="compare",
                )
                with gr.Row(elem_id="result-bottom"):
                    output_text = gr.Textbox(
                        lines=5,
                        label=TEXT["fa"]["ocr"],
                        interactive=False,
                        elem_id="ocr",
                        elem_classes=["rtl"],
                        scale=7,
                    )
                    with gr.Column(scale=3, elem_id="downloads"):
                        image_download = gr.DownloadButton(TEXT["fa"]["download_image"])
                        text_download = gr.DownloadButton(TEXT["fa"]["download_text"])

        with gr.Column(scale=5, elem_id="controls", elem_classes=["panel"]):
            gr.HTML(
                "<div><div class='upload-head'>تصویر فارسی را بارگذاری کنید</div>"
                "<div class='upload-sub'>PNG · JPG · JPEG · WEBP · BMP · TIFF</div></div>"
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
                "<div class='trust'>Lightweight AI SR x4 · OCR فارسی · بدون ارسال فایل</div>"
            )

    sample_button.click(fn=lambda: SAMPLE_SCAN, outputs=[input_image])
    process_button.click(
        fn=run_single,
        inputs=[input_image, profile, language],
        outputs=[comparison, output_text, image_download, text_download],
        show_progress="full",
    )
    language.change(
        fn=localize,
        inputs=[language],
        outputs=[
            header,
            input_image,
            sample_button,
            profile,
            process_button,
            comparison,
            output_text,
            image_download,
            text_download,
        ],
    )
    theme.click(fn=None, js=THEME_JS)

if __name__ == "__main__":
    print("[APP] starting fast one-screen workspace on http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
