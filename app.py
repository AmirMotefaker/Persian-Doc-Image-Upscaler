from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr

from persian_upscaler.service import process_image

PROFILE_CHOICES = {
    "fa": [("طبیعی", "طبیعی"), ("سند", "سند"), ("اسکن ضعیف", "اسکن ضعیف")],
    "en": [("Natural", "طبیعی"), ("Document", "سند"), ("Weak scan", "اسکن ضعیف")],
}

TEXT = {
    "fa": {
        "title": "دقیق‌خوان",
        "subtitle": "بهبود تصویر و استخراج دقیق متن فارسی",
        "status": "موتور OCR آماده است",
        "input_title": "۱. تصویر ورودی",
        "input_sub": "تصویر یا اسکن دارای متن فارسی را انتخاب کنید.",
        "file": "انتخاب تصویر",
        "processing_title": "۲. تنظیم پردازش",
        "processing_sub": "برای متن چاپی، حالت «سند» پیشنهاد می‌شود.",
        "profile": "نوع تصویر",
        "scale": "ضریب افزایش ابعاد",
        "button": "بهبود تصویر و استخراج متن",
        "hint": "طبیعی: عکس واضح · سند: متن چاپی · اسکن ضعیف: تصویر کم‌کیفیت",
        "result_title": "۳. نتیجه پردازش",
        "result_sub": "تصویر بهبودیافته، نمای OCR و متن استخراج‌شده را بررسی کنید.",
        "enhanced": "تصویر بهبودیافته",
        "ocr_view": "نمای مخصوص OCR",
        "text": "متن استخراج‌شده",
        "txt": "فایل متنی خروجی",
    },
    "en": {
        "title": "DaqiqKhan",
        "subtitle": "Persian image enhancement and accurate OCR",
        "status": "OCR engine ready",
        "input_title": "1. Input image",
        "input_sub": "Select an image or scan containing Persian text.",
        "file": "Select image",
        "processing_title": "2. Processing",
        "processing_sub": "Document mode is recommended for printed text.",
        "profile": "Image profile",
        "scale": "Upscale factor",
        "button": "Enhance image and extract text",
        "hint": "Natural: clean photo · Document: printed text · Weak scan: low-quality image",
        "result_title": "3. Results",
        "result_sub": "Review the enhanced image, OCR view and extracted text.",
        "enhanced": "Enhanced image",
        "ocr_view": "OCR preprocessing view",
        "text": "Extracted text",
        "txt": "TXT output",
    },
}


def _header(language: str) -> str:
    t = TEXT[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='topbar-content' dir='{direction}'>"
        "<div class='brand-wrap'><div class='brand-mark'>د</div><div>"
        f"<div class='brand-title'>{t['title']}</div>"
        f"<div class='brand-sub'>{t['subtitle']}</div></div></div>"
        f"<div class='status-chip'><span></span>{t['status']}</div>"
        "</div>"
    )


def _section(title: str, subtitle: str, language: str) -> str:
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='section-head' dir='{direction}'>"
        f"<strong>{title}</strong><span>{subtitle}</span></div>"
    )


def translate_ui(language: str):
    t = TEXT[language]
    text_class = ["ltr"] if language == "en" else ["rtl"]
    return (
        gr.HTML(value=_header(language)),
        gr.HTML(value=_section(t["input_title"], t["input_sub"], language)),
        gr.File(label=t["file"]),
        gr.HTML(value=_section(t["processing_title"], t["processing_sub"], language)),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Slider(label=t["scale"]),
        gr.Button(value=t["button"]),
        gr.HTML(value=f"<div class='hint'>{t['hint']}</div>"),
        gr.HTML(value=_section(t["result_title"], t["result_sub"], language)),
        gr.Image(label=t["enhanced"]),
        gr.Image(label=t["ocr_view"]),
        gr.Textbox(label=t["text"], elem_classes=text_class),
        gr.File(label=t["txt"]),
    )


def run(file_obj, profile, scale, language):
    t = TEXT[language]
    if file_obj is None:
        message = "Please select an image containing Persian text." if language == "en" else "لطفاً یک تصویر دارای متن فارسی انتخاب کنید."
        raise gr.Error(message)
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        raise gr.Error("Invalid input file." if language == "en" else "فایل ورودی معتبر نیست.")
    try:
        return process_image(path, profile, float(scale), language)
    except Exception as exc:
        prefix = "Processing failed" if language == "en" else "پردازش ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');
:root {color-scheme:dark;}
html, body {height:100%; margin:0; overflow:hidden !important;}
body, .gradio-container {font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif !important;}
.gradio-container {max-width:1560px !important; width:100% !important; height:100dvh !important; margin:0 auto !important; padding:10px 16px 12px !important; overflow:hidden !important; direction:rtl; box-sizing:border-box !important;}
body.english-mode .gradio-container {direction:ltr;}
footer, .footer, .built-with {display:none !important;}
#topbar {height:54px !important; min-height:54px !important; margin:0 0 8px !important;}
.topbar-content {height:54px; display:flex; align-items:center; justify-content:space-between; gap:14px;}
.brand-wrap {display:flex; align-items:center; gap:10px; min-width:0;}
.brand-mark {width:38px; height:38px; flex:0 0 38px; border-radius:12px; display:grid; place-items:center; font-size:20px; font-weight:800; color:white; background:linear-gradient(145deg,#6366f1,#8b5cf6); box-shadow:0 8px 24px rgba(99,102,241,.25);}
.brand-title {font-size:1.04rem; font-weight:800; line-height:1.25;}
.brand-sub {font-size:.72rem; opacity:.55; margin-top:2px; white-space:nowrap;}
.status-chip {display:inline-flex; align-items:center; gap:7px; border:1px solid rgba(34,197,94,.22); background:rgba(34,197,94,.07); padding:6px 10px; border-radius:999px; font-size:.72rem; white-space:nowrap;}
.status-chip span {width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,.11);}
#workspace {display:flex !important; flex-wrap:nowrap !important; width:100% !important; height:calc(100dvh - 84px) !important; gap:12px !important; align-items:stretch !important;}
#controls-panel {flex:0 0 34% !important; width:34% !important; max-width:34% !important; min-width:0 !important;}
#output-panel {flex:1 1 66% !important; width:66% !important; max-width:66% !important; min-width:0 !important;}
.panel {height:100% !important; min-height:0 !important; overflow:hidden !important; border:1px solid var(--border-color-primary) !important; border-radius:18px !important; padding:12px !important; box-sizing:border-box !important; background:var(--background-fill-secondary) !important;}
#controls-panel {display:flex !important; flex-direction:column !important; gap:7px !important;}
#output-panel {display:flex !important; flex-direction:column !important;}
#utility-row {margin:0 0 2px !important; gap:7px !important; align-items:center !important;}
#language {min-width:0 !important; flex:1 1 auto !important;}
#theme-toggle {flex:0 0 46px !important; width:46px !important; min-width:46px !important;}
#theme-toggle button {height:38px !important; min-height:38px !important; border-radius:11px !important; font-size:16px !important;}
.section-head {margin:1px 0 5px;}
.section-head strong {display:block; font-size:.91rem; font-weight:800; line-height:1.35;}
.section-head span {display:block; opacity:.5; font-size:.69rem; margin-top:2px; line-height:1.5;}
#upload-box {min-height:0 !important; max-height:125px !important; overflow:hidden !important;}
#upload-box > div {min-height:105px !important;}
#profile, #scale {margin:0 !important;}
#process-button button {height:44px !important; min-height:44px !important; border-radius:12px !important; font-weight:800 !important; font-size:.88rem !important;}
.hint {font-size:.69rem; opacity:.58; line-height:1.65; margin-top:1px;}
#results-head {flex:0 0 auto !important;}
#results-tabs {flex:1 1 auto !important; min-height:0 !important; height:100% !important; overflow:hidden !important;}
#results-tabs > div {height:100% !important; min-height:0 !important;}
.result-image {height:calc(100dvh - 190px) !important; min-height:300px !important;}
.result-text textarea {height:calc(100dvh - 285px) !important; min-height:230px !important;}
.rtl, .rtl textarea, .rtl input {direction:rtl !important; text-align:right !important;}
.ltr, .ltr textarea, .ltr input {direction:ltr !important; text-align:left !important; font-family:'Inter',sans-serif !important;}
body.light-mode {color-scheme:light; background:#f5f7fb !important;}
body.light-mode .gradio-container {background:#f5f7fb !important; color:#101828 !important;}
body.light-mode .panel {background:#fff !important; border-color:#e4e7ec !important;}
body.light-mode .status-chip {color:#157f3b; background:#edfdf2; border-color:#b7ebc6;}
@media (max-width:950px) {
  html, body {overflow:auto !important;}
  .gradio-container {height:auto !important; min-height:100dvh !important; overflow:visible !important; padding:10px !important;}
  #workspace {height:auto !important; flex-wrap:wrap !important;}
  #controls-panel, #output-panel {flex:1 1 100% !important; width:100% !important; max-width:100% !important;}
  .panel {height:auto !important; overflow:visible !important;}
  .result-image {height:420px !important;}
}
"""

INIT_JS = """
() => {
  const saved = localStorage.getItem('dq-theme') || 'dark';
  document.body.classList.toggle('light-mode', saved === 'light');
}
"""

THEME_JS = """
() => {
  const light = !document.body.classList.contains('light-mode');
  document.body.classList.toggle('light-mode', light);
  localStorage.setItem('dq-theme', light ? 'light' : 'dark');
}
"""

LANG_JS = """
(lang) => {
  document.body.classList.toggle('english-mode', lang === 'en');
  document.documentElement.lang = lang === 'en' ? 'en' : 'fa';
  document.documentElement.dir = lang === 'en' ? 'ltr' : 'rtl';
  return lang;
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(_header("fa"), elem_id="topbar")

    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_id="controls-panel", elem_classes=["panel"]):
            with gr.Row(elem_id="utility-row"):
                language = gr.Radio(
                    [("فارسی", "fa"), ("English", "en")],
                    value="fa",
                    label=None,
                    elem_id="language",
                )
                theme = gr.Button("☼", elem_id="theme-toggle")

            input_head = gr.HTML(_section(TEXT["fa"]["input_title"], TEXT["fa"]["input_sub"], "fa"))
            input_file = gr.File(
                label=TEXT["fa"]["file"],
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
                elem_id="upload-box",
            )

            processing_head = gr.HTML(_section(TEXT["fa"]["processing_title"], TEXT["fa"]["processing_sub"], "fa"))
            profile = gr.Radio(
                PROFILE_CHOICES["fa"],
                value="سند",
                label=TEXT["fa"]["profile"],
                elem_id="profile",
            )
            scale = gr.Slider(
                1.0,
                3.0,
                value=2.0,
                step=0.5,
                label=TEXT["fa"]["scale"],
                elem_id="scale",
            )
            submit = gr.Button(
                TEXT["fa"]["button"],
                variant="primary",
                elem_id="process-button",
            )
            hint = gr.HTML(f"<div class='hint'>{TEXT['fa']['hint']}</div>")

        with gr.Column(elem_id="output-panel", elem_classes=["panel"]):
            results_head = gr.HTML(
                _section(TEXT["fa"]["result_title"], TEXT["fa"]["result_sub"], "fa"),
                elem_id="results-head",
            )
            with gr.Tabs(elem_id="results-tabs"):
                with gr.Tab("◫"):
                    output_image = gr.Image(
                        type="filepath",
                        label=TEXT["fa"]["enhanced"],
                        elem_classes=["result-image"],
                    )
                with gr.Tab("OCR"):
                    ocr_preview = gr.Image(
                        type="filepath",
                        label=TEXT["fa"]["ocr_view"],
                        elem_classes=["result-image"],
                    )
                with gr.Tab("Aa"):
                    output_text = gr.Textbox(
                        lines=12,
                        label=TEXT["fa"]["text"],
                        buttons=["copy"],
                        elem_classes=["rtl", "result-text"],
                    )
                    text_file = gr.File(label=TEXT["fa"]["txt"])

    language.change(
        fn=translate_ui,
        inputs=[language],
        outputs=[
            header,
            input_head,
            input_file,
            processing_head,
            profile,
            scale,
            submit,
            hint,
            results_head,
            output_image,
            ocr_preview,
            output_text,
            text_file,
        ],
        js=LANG_JS,
    )
    theme.click(fn=None, js=THEME_JS)
    submit.click(
        fn=run,
        inputs=[input_file, profile, scale, language],
        outputs=[output_image, ocr_preview, output_text, text_file],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS, js=INIT_JS)
