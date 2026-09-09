from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
from PIL import Image

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
        "hero_sub": "برای اسناد، اسکن‌ها، اسکرین‌شات‌ها و تصاویر دارای متن فارسی.",
        "single": "تکی",
        "batch": "گروهی",
        "upload": "تصویر را اینجا رها کنید یا کلیک کنید",
        "batch_upload": "۲ تا ۲۰ تصویر انتخاب کنید",
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
        "fast": "سریع",
        "persian": "ویژه فارسی",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian Image Enhancement & OCR",
        "hero": "Make Persian text clearer and extract it accurately",
        "hero_sub": "Built for documents, scans, screenshots and Persian text-heavy images.",
        "single": "Single",
        "batch": "Batch",
        "upload": "Drop an image here or click to browse",
        "batch_upload": "Select 2 to 20 images",
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
        "fast": "Fast",
        "persian": "Persian-first",
    },
}


def _hero(language: str) -> str:
    t = COPY[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='hero' dir='{direction}'>"
        f"<h1>{t['hero']}</h1>"
        f"<p>{t['hero_sub']}</p>"
        "</div>"
    )


def _brand(language: str) -> str:
    t = COPY[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='brandbar' dir='{direction}'>"
        "<div class='brand-left'><div class='logo'>د</div><div>"
        f"<div class='brand-name'>{t['brand']}</div>"
        f"<div class='brand-tag'>{t['tagline']}</div></div></div>"
        f"<div class='engine'><span></span>{t['ready']}</div>"
        "</div>"
    )


def _badges(language: str) -> str:
    t = COPY[language]
    return (
        "<div class='badges'>"
        f"<span>✓ {t['secure']}</span>"
        f"<span>⚡ {t['fast']}</span>"
        f"<span>فا {t['persian']}</span>"
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
html, body {margin:0; min-height:100%; background:#0d1220;}
body, .gradio-container {font-family:'Vazirmatn','Inter',sans-serif !important;}
.gradio-container {max-width:1500px !important; margin:0 auto !important; padding:14px 22px 26px !important;}
footer, .footer, .built-with {display:none !important;}
#brand {margin-bottom:8px !important;}
.brandbar {height:56px; display:flex; align-items:center; justify-content:space-between; gap:16px;}
.brand-left {display:flex; align-items:center; gap:11px;}
.logo {width:42px; height:42px; border-radius:13px; display:grid; place-items:center; font-size:22px; font-weight:800; color:white; background:linear-gradient(135deg,#7c3aed,#db2777); box-shadow:0 10px 28px rgba(124,58,237,.28);}
.brand-name {font-size:1.12rem; font-weight:800;}
.brand-tag {font-size:.72rem; opacity:.55; margin-top:2px;}
.engine {display:flex; align-items:center; gap:7px; border:1px solid rgba(34,197,94,.22); background:rgba(34,197,94,.07); padding:7px 11px; border-radius:999px; font-size:.72rem;}
.engine span {width:7px; height:7px; border-radius:50%; background:#22c55e;}
.hero {text-align:center; padding:8px 0 15px;}
.hero h1 {font-size:2rem; line-height:1.45; margin:0; font-weight:800; letter-spacing:-.025em; background:linear-gradient(90deg,#a78bfa,#ec4899); -webkit-background-clip:text; color:transparent;}
.hero p {font-size:.9rem; opacity:.62; margin:6px 0 0;}
#workspace {gap:18px !important; align-items:stretch !important;}
#preview-card {border:1px solid rgba(148,163,184,.14) !important; background:#111827 !important; border-radius:22px !important; padding:14px !important; min-height:650px !important;}
#controls-card {border:1px solid rgba(148,163,184,.14) !important; background:#111827 !important; border-radius:22px !important; padding:16px !important; min-height:650px !important;}
#compare {height:455px !important; min-height:455px !important;}
#compare > div {height:100% !important;}
#input-preview {height:265px !important; min-height:265px !important; border:1px dashed rgba(167,139,250,.38) !important; border-radius:16px !important; overflow:hidden !important;}
#input-preview img {object-fit:contain !important;}
#batch-upload {min-height:150px !important;}
#mode, #profile, #format {margin-bottom:6px !important;}
#single-action button, #batch-action button {min-height:48px !important; border-radius:13px !important; font-weight:800 !important; background:linear-gradient(90deg,#7c3aed,#db2777) !important; border:none !important;}
#result-row {gap:10px !important; margin-top:10px !important;}
#ocr-preview {height:170px !important; min-height:170px !important;}
#ocr-text textarea {min-height:138px !important; resize:none !important;}
.badges {display:flex; justify-content:center; gap:18px; flex-wrap:wrap; padding-top:10px; font-size:.71rem; opacity:.58;}
#downloads {gap:8px !important; margin-top:6px !important;}
@media(max-width:950px){.hero h1{font-size:1.55rem}.gradio-container{padding:10px !important}#preview-card,#controls-card{min-height:auto !important}#compare{height:360px !important;min-height:360px !important}}
"""

THEME_JS = """
() => {
  const root = document.documentElement;
  const light = root.dataset.theme !== 'light';
  root.dataset.theme = light ? 'light' : 'dark';
  document.body.style.background = light ? '#f3f5fa' : '#0d1220';
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    brand = gr.HTML(_brand("fa"), elem_id="brand")
    hero = gr.HTML(_hero("fa"))

    with gr.Row():
        language = gr.Radio([("فارسی", "fa"), ("English", "en")], value="fa", label=None, scale=3)
        theme = gr.Button("☼", scale=1)

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=7, elem_id="preview-card"):
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
                    lines=6,
                    label=COPY["fa"]["text"],
                    elem_id="ocr-text",
                )
            with gr.Row(elem_id="downloads"):
                enhanced_file = gr.File(label=COPY["fa"]["download"])
                text_file = gr.File(label=COPY["fa"]["download_text"])

        with gr.Column(scale=5, elem_id="controls-card"):
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
            )
            single_action = gr.Button(COPY["fa"]["process"], variant="primary", elem_id="single-action")
            batch_action = gr.Button(COPY["fa"]["process_batch"], variant="primary", visible=False, elem_id="batch-action")
            badges = gr.HTML(_badges("fa"))
            batch_report = gr.Textbox(lines=7, label=COPY["fa"]["batch_result"], visible=False)
            batch_zip = gr.File(label=COPY["fa"]["batch_zip"], visible=False)

    mode.change(
        fn=switch_mode,
        inputs=[mode],
        outputs=[single_box, batch_box],
    ).then(
        fn=lambda value: (gr.update(visible=value == "single"), gr.update(visible=value == "batch"), gr.update(visible=value == "batch"), gr.update(visible=value == "batch")),
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
