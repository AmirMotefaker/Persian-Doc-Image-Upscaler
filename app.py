from __future__ import annotations

# Gradio callback injection and embedded CSS intentionally use patterns that conflict
# with generic Ruff style rules but are valid for this UI entrypoint.
# ruff: noqa: E402, E501, B008

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr
from PIL import Image

from persian_upscaler.service import process_image

PROFILE_CHOICES = {
    "fa": [("سند", "سند"), ("اسکرین‌شات", "طبیعی"), ("اسکن ضعیف", "اسکن ضعیف")],
    "en": [("Document", "سند"), ("Screenshot", "طبیعی"), ("Weak scan", "اسکن ضعیف")],
}

TEXT = {
    "fa": {
        "brand": "دقیق‌خوان",
        "tagline": "افزایش کیفیت تصویر و OCR تخصصی فارسی",
        "headline": "تصویر فارسی را واضح‌تر کن؛ متن دقیق تحویل بگیر",
        "upload": "تصویر را اینجا رها کنید یا کلیک کنید",
        "sample": "نمونه",
        "profile": "نوع تصویر",
        "action": "افزایش کیفیت و استخراج متن",
        "compare": "قبل / بعد",
        "ocr": "متن استخراج‌شده",
        "download_image": "دریافت تصویر",
        "download_text": "دریافت متن",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian image enhancement and OCR",
        "headline": "Make Persian images clearer and extract accurate text",
        "upload": "Drop an image here or click to upload",
        "sample": "Sample",
        "profile": "Image type",
        "action": "Enhance & extract text",
        "compare": "Before / After",
        "ocr": "Extracted text",
        "download_image": "Download image",
        "download_text": "Download text",
    },
}


def _make_sample() -> str:
    workdir = Path(tempfile.mkdtemp(prefix="daqiqkhan-sample-"))
    path = workdir / "sample.png"
    canvas = Image.new("RGB", (960, 540), "white")
    canvas.save(path, format="PNG")
    return str(path)


SAMPLE_IMAGE = _make_sample()


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
        f"<div class='topbar' dir='{direction}'>"
        "<div class='brandmark'>د</div>"
        f"<div class='brandcopy'><strong>{t['brand']}</strong>"
        f"<span>{t['tagline']}</span></div>"
        f"<div class='headline'>{t['headline']}</div>"
        "</div>"
    )


def run_single(image_path, profile, language, progress=gr.Progress()):
    if not image_path:
        raise gr.Error(
            "Please upload an image." if language == "en" else "لطفاً یک تصویر بارگذاری کنید."
        )

    progress(0.08, desc="در حال آماده‌سازی")
    try:
        enhanced, _ocr_preview, canonical_text, text_file = process_image(
            str(image_path),
            profile=profile,
            scale=4.0,
            language=language,
            output_format="PNG",
            engine="Super-Resolution Pro",
        )
        progress(0.92, desc="در حال آماده‌سازی خروجی")
        comparison = _comparison_pair(str(image_path), enhanced)
        progress(1.0, desc="انجام شد")
        return comparison, canonical_text, enhanced, text_file
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


def localize(language: str):
    t = TEXT[language]
    text_class = ["ltr"] if language == "en" else ["rtl"]
    return (
        gr.HTML(value=_header(language)),
        gr.Image(label=t["upload"]),
        gr.Button(value=t["sample"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Button(value=t["action"]),
        gr.ImageSlider(label=t["compare"]),
        gr.Textbox(label=t["ocr"], elem_classes=text_class),
        gr.DownloadButton(label=t["download_image"]),
        gr.DownloadButton(label=t["download_text"]),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
:root{--bg:#08111f;--panel:#0e1a2a;--panel2:#111f32;--line:#22334b;--text:#f8fafc;--muted:#8fa1b8;--accent:#22c7d9;--accent2:#5b8cff}
*{box-sizing:border-box!important}
html,body{height:100%;margin:0;overflow:hidden!important;background:var(--bg)!important;color:var(--text)!important}
body,.gradio-container{font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif!important}
.gradio-container{max-width:1460px!important;width:100%!important;height:100dvh!important;margin:0 auto!important;padding:10px 14px!important;overflow:hidden!important;background:var(--bg)!important}
footer,.footer,.built-with{display:none!important}
#header{height:54px!important;min-height:54px!important;margin:0 0 8px!important}
.topbar{height:54px;display:grid;grid-template-columns:auto 220px 1fr;align-items:center;gap:12px;direction:rtl}
.brandmark{width:36px;height:36px;display:grid;place-items:center;border-radius:11px;background:linear-gradient(135deg,var(--accent2),var(--accent));font-weight:800}
.brandcopy strong{display:block;font-size:.98rem}.brandcopy span{display:block;color:var(--muted);font-size:.58rem;margin-top:1px}.headline{text-align:left;font-size:1.03rem;font-weight:800;color:#dff8ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#workspace{height:calc(100dvh - 82px)!important;min-height:0!important;gap:14px!important;overflow:hidden!important;align-items:stretch!important}
.panel{height:100%!important;min-height:0!important;overflow:hidden!important}
#result-shell{height:100%!important;display:grid!important;grid-template-rows:minmax(0,1fr) 142px!important;gap:8px!important;background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:20px!important;padding:10px!important}
#compare{height:100%!important;min-height:0!important;border-radius:15px!important;overflow:hidden!important;background:var(--panel2)!important}
#compare>div{height:100%!important;min-height:0!important}
#result-bottom{height:142px!important;gap:8px!important;margin:0!important}
#ocr{height:142px!important;min-height:142px!important}
#ocr textarea{height:100px!important;min-height:100px!important;resize:none!important;font-size:.76rem!important;line-height:1.7!important}
.rtl textarea{direction:rtl!important;text-align:right!important}.ltr textarea{direction:ltr!important;text-align:left!important;font-family:'Inter',sans-serif!important}
#downloads{height:142px!important;display:grid!important;grid-template-rows:1fr 1fr!important;gap:7px!important}
#downloads button{height:100%!important;border-radius:12px!important;font-weight:700!important}
#controls{height:100%!important;display:grid!important;grid-template-rows:auto minmax(0,1fr) 36px 72px 62px auto!important;gap:9px!important;background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:20px!important;padding:14px!important;overflow:hidden!important}
.upload-title{text-align:center;font-weight:800;font-size:1rem}.upload-sub{text-align:center;color:var(--muted);font-size:.62rem;margin-top:2px}
#input{height:100%!important;min-height:0!important;border:1.5px dashed #2d6b82!important;border-radius:16px!important;overflow:hidden!important;background:var(--panel2)!important}
#input>div{height:100%!important;min-height:0!important}#input img{object-fit:contain!important}
#sample button{height:36px!important;min-height:36px!important;border-radius:10px!important}
#profile{height:72px!important;min-height:72px!important;margin:0!important}
#action button{height:62px!important;min-height:62px!important;border-radius:14px!important;border:none!important;background:linear-gradient(90deg,var(--accent2),var(--accent))!important;font-size:1rem!important;font-weight:800!important;box-shadow:0 12px 28px rgba(34,199,217,.16)!important}
.trust{text-align:center;color:var(--muted);font-size:.60rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#lang{position:fixed!important;top:14px!important;left:50%!important;transform:translateX(-50%)!important;width:126px!important;z-index:30!important}
#lang input{height:34px!important;min-height:34px!important}
@media(max-height:820px){#header{height:46px!important;min-height:46px!important}.topbar{height:46px}.brandcopy span{display:none}#workspace{height:calc(100dvh - 68px)!important}#result-shell{grid-template-rows:minmax(0,1fr) 118px!important}#result-bottom,#downloads,#ocr{height:118px!important;min-height:118px!important}#ocr textarea{height:78px!important;min-height:78px!important}#controls{grid-template-rows:auto minmax(0,1fr) 32px 64px 54px auto!important}#action button{height:54px!important;min-height:54px!important}}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(_header("fa"), elem_id="header")
    language = gr.Dropdown(
        choices=[("فارسی", "fa"), ("English", "en")],
        value="fa",
        label=None,
        show_label=False,
        elem_id="lang",
    )

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
                "<div><div class='upload-title'>تصویر فارسی را بارگذاری کنید</div>"
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
                "<div class='trust'>OCR فارسی V2 · حفظ جدول · پردازش امن سند</div>"
            )

    sample_button.click(fn=lambda: SAMPLE_IMAGE, outputs=[input_image])
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

if __name__ == "__main__":
    print("[APP] starting Persian OCR V2 on http://127.0.0.1:7860", flush=True)
    demo.queue(default_concurrency_limit=1).launch(css=CSS)
