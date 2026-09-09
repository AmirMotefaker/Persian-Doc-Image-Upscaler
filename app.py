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

from persian_upscaler.service import process_batch, process_image

PROFILE_CHOICES = {
    "fa": [("سند", "سند"), ("تصویر واضح", "طبیعی"), ("اسکن ضعیف", "اسکن ضعیف")],
    "en": [("Document", "سند"), ("Clean image", "طبیعی"), ("Weak scan", "اسکن ضعیف")],
}

COPY = {
    "fa": {
        "brand": "دقیق‌خوان",
        "tagline": "بهبود کیفیت تصویر و OCR تخصصی فارسی",
        "hero": "متن فارسی را واضح‌تر کنید، دقیق‌تر استخراج کنید",
        "hero_sub": "اسناد، اسکن‌ها، اسکرین‌شات‌ها و تصاویر دارای متن فارسی",
        "single": "تکی",
        "batch": "گروهی",
        "upload": "تصویر را اینجا رها کنید یا کلیک کنید",
        "batch_upload": "۲ تا ۲۰ تصویر انتخاب کنید",
        "sample": "نمونه اسکن فارسی",
        "model": "مدل پردازش",
        "format": "فرمت خروجی",
        "scale": "افزایش ابعاد",
        "process": "بهبود تصویر و استخراج متن",
        "process_batch": "پردازش گروهی",
        "compare": "قبل / بعد",
        "ocr": "نمای OCR",
        "text": "متن استخراج‌شده",
        "download": "دریافت تصویر",
        "download_text": "دریافت متن",
        "batch_result": "گزارش پردازش گروهی",
        "batch_zip": "دریافت فایل ZIP",
        "ready": "موتور فارسی آماده",
        "secure": "پردازش امن",
        "fast": "چندمرحله‌ای",
        "persian": "ویژه فارسی",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian Image Enhancement & OCR",
        "hero": "Make Persian text clearer and extract it accurately",
        "hero_sub": "Documents, scans, screenshots and Persian text-heavy images",
        "single": "Single",
        "batch": "Batch",
        "upload": "Drop an image here or click to browse",
        "batch_upload": "Select 2 to 20 images",
        "sample": "Persian scan sample",
        "model": "Processing model",
        "format": "Output format",
        "scale": "Upscale factor",
        "process": "Enhance image & extract text",
        "process_batch": "Process batch",
        "compare": "Before / After",
        "ocr": "OCR view",
        "text": "Extracted text",
        "download": "Download image",
        "download_text": "Download text",
        "batch_result": "Batch report",
        "batch_zip": "Download ZIP",
        "ready": "Persian engine ready",
        "secure": "Secure",
        "fast": "Multi-pass",
        "persian": "Persian-first",
    },
}


def _hero(language: str) -> str:
    t = COPY[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='hero' dir='{direction}'>"
        f"<h1>{t['hero']}</h1><p>{t['hero_sub']}</p></div>"
    )


def _brand(language: str) -> str:
    t = COPY[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='brandbar' dir='{direction}'>"
        "<div class='brand-left'><div class='logo'>د</div><div>"
        f"<div class='brand-name'>{t['brand']}</div>"
        f"<div class='brand-tag'>{t['tagline']}</div></div></div>"
        f"<div class='engine'><span></span>{t['ready']}</div></div>"
    )


def _badges(language: str) -> str:
    t = COPY[language]
    return (
        "<div class='badges'>"
        f"<span>✓ {t['secure']}</span>"
        f"<span>◎ {t['fast']}</span>"
        f"<span>فا {t['persian']}</span></div>"
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
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _make_scan_sample() -> str:
    workdir = Path(tempfile.mkdtemp(prefix="daqiqkhan-sample-"))
    path = workdir / "persian-scan-sample.png"
    canvas = Image.new("L", (1180, 760), 242)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(38)
    text_font = _font(27)
    small_font = _font(22)

    lines = [
        "نمونه سند فارسی برای ارزیابی کیفیت OCR",
        "شماره سند: ۱۴۰۵-۰۶-۱۸     تاریخ ثبت: ۱۴۰۵/۰۶/۱۸",
        "نام کالا: گواهی سپرده کالایی",
        "مقدار: ۲۷٬۰۰۰٬۰۰۰ ریال",
        "توضیحات: این تصویر عمداً شبیه یک اسکن کم‌کیفیت ساخته شده است.",
        "هدف: بررسی حفظ نقطه‌ها، دندانه‌ها، اعداد و فاصله‌های فارسی.",
    ]
    y = 85
    for index, line in enumerate(lines):
        font = title_font if index == 0 else text_font
        try:
            draw.text((1080, y), line, fill=28, font=font, anchor="ra", direction="rtl")
        except (TypeError, ValueError):
            draw.text((80, y), line, fill=28, font=font)
        y += 92 if index == 0 else 78

    draw.rectangle((65, 610, 1115, 700), outline=105, width=2)
    try:
        draw.text((1080, 642), "یادداشت: خروجی باید خوانا، مرتب و قابل کپی باشد.", fill=55, font=small_font, anchor="ra", direction="rtl")
    except (TypeError, ValueError):
        draw.text((90, 642), "Persian OCR sample document", fill=55, font=small_font)

    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.55))
    canvas = canvas.rotate(0.35, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=245)
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
        comparison = _comparison_pair(str(image_path), enhanced)
        return comparison, ocr_preview, summary, enhanced, text_file
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def run_batch(files, profile, scale, output_format, language):
    paths = [str(item) for item in (files or [])]
    if len(paths) < 2:
        raise gr.Error(
            "Please select at least 2 images."
            if language == "en"
            else "برای پردازش گروهی حداقل ۲ تصویر انتخاب کنید."
        )
    try:
        return process_batch(
            paths,
            profile=profile,
            scale=float(scale),
            language=language,
            output_format=output_format,
        )
    except Exception as exc:
        prefix = "Batch processing failed" if language == "en" else "پردازش گروهی ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def switch_mode(mode):
    is_batch = mode == "batch"
    return gr.update(visible=not is_batch), gr.update(visible=is_batch)


def localize(language):
    t = COPY[language]
    return (
        gr.HTML(_brand(language)),
        gr.HTML(_hero(language)),
        gr.Radio(choices=[(t["single"], "single"), (t["batch"], "batch")]),
        gr.Image(label=t["upload"]),
        gr.File(label=t["batch_upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["model"]),
        gr.Radio(label=t["format"]),
        gr.Slider(label=t["scale"]),
        gr.Button(value=t["process"]),
        gr.Button(value=t["process_batch"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Image(label=t["ocr"]),
        gr.Textbox(label=t["text"]),
        gr.File(label=t["download"]),
        gr.File(label=t["download_text"]),
        gr.Textbox(label=t["batch_result"]),
        gr.File(label=t["batch_zip"]),
        gr.HTML(_badges(language)),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
html, body {margin:0; height:100%; overflow:hidden !important; background:#0b1020;}
body, .gradio-container {font-family:'Vazirmatn','Inter',sans-serif !important;}
.gradio-container {max-width:1500px !important; width:100% !important; height:100dvh !important; margin:0 auto !important; padding:7px 16px 9px !important; overflow:hidden !important; box-sizing:border-box !important;}
footer, .footer, .built-with {display:none !important;}
#brand {height:46px !important; margin:0 0 3px !important; overflow:hidden !important;}
.brandbar {height:46px; display:flex; align-items:center; justify-content:space-between; gap:14px;}
.brand-left {display:flex; align-items:center; gap:9px;}
.logo {width:34px; height:34px; border-radius:11px; display:grid; place-items:center; font-size:18px; font-weight:800; color:white; background:linear-gradient(135deg,#7c3aed,#db2777); box-shadow:0 8px 22px rgba(124,58,237,.25);}
.brand-name {font-size:.98rem; font-weight:800; line-height:1.15;}
.brand-tag {font-size:.64rem; opacity:.52; margin-top:1px;}
.engine {display:flex; align-items:center; gap:6px; border:1px solid rgba(34,197,94,.22); background:rgba(34,197,94,.07); padding:5px 9px; border-radius:999px; font-size:.64rem;}
.engine span {width:6px; height:6px; border-radius:50%; background:#22c55e;}
.hero {text-align:center; height:56px; padding:2px 0 4px; box-sizing:border-box; overflow:hidden;}
.hero h1 {font-size:1.5rem; line-height:1.45; margin:0; font-weight:800; letter-spacing:-.02em; background:linear-gradient(90deg,#a78bfa,#ec4899); -webkit-background-clip:text; color:transparent;}
.hero p {font-size:.72rem; opacity:.57; margin:2px 0 0;}
#utility-row {height:34px !important; min-height:34px !important; margin:0 0 5px !important; gap:7px !important; align-items:center !important;}
#utility-row > div {min-height:0 !important;}
#workspace {display:flex !important; gap:11px !important; height:calc(100dvh - 160px) !important; min-height:0 !important; align-items:stretch !important; overflow:hidden !important;}
#preview-card {flex:1 1 68% !important; min-width:0 !important;}
#controls-card {flex:0 0 31% !important; min-width:330px !important; max-width:31% !important;}
.card {height:100% !important; min-height:0 !important; overflow:hidden !important; border:1px solid rgba(148,163,184,.14) !important; background:#101827 !important; border-radius:18px !important; padding:10px !important; box-sizing:border-box !important;}
#preview-card {display:flex !important; flex-direction:column !important; gap:7px !important;}
#compare {flex:1 1 65% !important; min-height:0 !important; height:auto !important;}
#compare > div {height:100% !important; min-height:0 !important;}
#result-row {flex:0 0 27% !important; min-height:0 !important; gap:7px !important; margin:0 !important;}
#ocr-preview {height:100% !important; min-height:0 !important;}
#ocr-preview > div {height:100% !important;}
#ocr-text {height:100% !important; min-height:0 !important;}
#ocr-text textarea {height:calc(100% - 29px) !important; min-height:0 !important; resize:none !important;}
#downloads {flex:0 0 46px !important; min-height:0 !important; gap:7px !important; margin:0 !important;}
#downloads > div {min-height:40px !important;}
#controls-card {display:flex !important; flex-direction:column !important; gap:5px !important;}
#mode {margin:0 !important;}
#input-preview {height:155px !important; min-height:155px !important; max-height:155px !important; border:1px dashed rgba(167,139,250,.38) !important; border-radius:14px !important; overflow:hidden !important;}
#input-preview img {object-fit:contain !important;}
#sample-button button {height:32px !important; min-height:32px !important; font-size:.7rem !important; border-radius:9px !important;}
#batch-upload {height:145px !important; min-height:145px !important; max-height:145px !important; overflow:hidden !important;}
#profile, #format, #scale {margin:0 !important;}
#single-action button, #batch-action button {min-height:42px !important; height:42px !important; border-radius:11px !important; font-weight:800 !important; font-size:.82rem !important; background:linear-gradient(90deg,#7c3aed,#db2777) !important; border:none !important;}
.badges {display:flex; justify-content:center; gap:12px; flex-wrap:wrap; padding-top:3px; font-size:.63rem; opacity:.56;}
@media(max-width:980px){html,body{overflow:auto !important}.gradio-container{height:auto !important;overflow:visible !important;padding:9px !important}#workspace{height:auto !important;flex-wrap:wrap !important}#preview-card,#controls-card{flex:1 1 100% !important;max-width:100% !important;min-width:0 !important;height:auto !important;overflow:visible !important}.card{height:auto !important;overflow:visible !important}#compare{height:390px !important}#result-row{height:260px !important}#input-preview{height:220px !important;max-height:220px !important}}
"""

THEME_JS = """
() => {
  const light = document.body.dataset.theme !== 'light';
  document.body.dataset.theme = light ? 'light' : 'dark';
  document.body.style.background = light ? '#f3f5fa' : '#0b1020';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    brand = gr.HTML(_brand("fa"), elem_id="brand")
    hero = gr.HTML(_hero("fa"))

    with gr.Row(elem_id="utility-row"):
        language = gr.Radio([("فارسی", "fa"), ("English", "en")], value="fa", label=None, scale=5)
        theme = gr.Button("☼", scale=1)

    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_id="preview-card", elem_classes=["card"]):
            comparison = gr.ImageSlider(
                label=COPY["fa"]["compare"],
                type="filepath",
                interactive=False,
                elem_id="compare",
            )
            with gr.Row(elem_id="result-row"):
                ocr_preview = gr.Image(
                    type="filepath",
                    label=COPY["fa"]["ocr"],
                    interactive=False,
                    elem_id="ocr-preview",
                )
                output_text = gr.Textbox(
                    lines=5,
                    label=COPY["fa"]["text"],
                    elem_id="ocr-text",
                )
            with gr.Row(elem_id="downloads"):
                enhanced_file = gr.File(label=COPY["fa"]["download"])
                text_file = gr.File(label=COPY["fa"]["download_text"])

        with gr.Column(elem_id="controls-card", elem_classes=["card"]):
            mode = gr.Radio(
                [(COPY["fa"]["single"], "single"), (COPY["fa"]["batch"], "batch")],
                value="single",
                label=None,
                elem_id="mode",
            )

            with gr.Column(visible=True) as single_box:
                input_image = gr.Image(
                    type="filepath",
                    sources=["upload", "clipboard"],
                    label=COPY["fa"]["upload"],
                    elem_id="input-preview",
                )
                sample_button = gr.Button(COPY["fa"]["sample"], elem_id="sample-button")

            with gr.Column(visible=False) as batch_box:
                batch_files = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    label=COPY["fa"]["batch_upload"],
                    elem_id="batch-upload",
                )

            profile = gr.Radio(
                PROFILE_CHOICES["fa"],
                value="سند",
                label=COPY["fa"]["model"],
                elem_id="profile",
            )
            output_format = gr.Radio(
                ["PNG", "JPG", "WEBP"],
                value="PNG",
                label=COPY["fa"]["format"],
                elem_id="format",
            )
            scale = gr.Slider(
                1.0,
                3.0,
                value=2.0,
                step=0.5,
                label=COPY["fa"]["scale"],
                elem_id="scale",
            )
            single_action = gr.Button(COPY["fa"]["process"], variant="primary", elem_id="single-action")
            batch_action = gr.Button(COPY["fa"]["process_batch"], variant="primary", visible=False, elem_id="batch-action")
            badges = gr.HTML(_badges("fa"))
            batch_report = gr.Textbox(lines=5, label=COPY["fa"]["batch_result"], visible=False)
            batch_zip = gr.File(label=COPY["fa"]["batch_zip"], visible=False)

    mode.change(
        fn=switch_mode,
        inputs=[mode],
        outputs=[single_box, batch_box],
    ).then(
        fn=lambda value: (
            gr.update(visible=value == "single"),
            gr.update(visible=value == "batch"),
            gr.update(visible=value == "batch"),
            gr.update(visible=value == "batch"),
        ),
        inputs=[mode],
        outputs=[single_action, batch_action, batch_report, batch_zip],
    )

    language.change(
        fn=localize,
        inputs=[language],
        outputs=[
            brand,
            hero,
            mode,
            input_image,
            batch_files,
            sample_button,
            profile,
            output_format,
            scale,
            single_action,
            batch_action,
            comparison,
            ocr_preview,
            output_text,
            enhanced_file,
            text_file,
            batch_report,
            batch_zip,
            badges,
        ],
    )
    theme.click(fn=None, js=THEME_JS)
    sample_button.click(fn=lambda: SAMPLE_SCAN, outputs=[input_image])

    single_action.click(
        fn=run_single,
        inputs=[input_image, profile, scale, output_format, language],
        outputs=[comparison, ocr_preview, output_text, enhanced_file, text_file],
    )
    batch_action.click(
        fn=run_batch,
        inputs=[batch_files, profile, scale, output_format, language],
        outputs=[batch_zip, batch_report],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS)
