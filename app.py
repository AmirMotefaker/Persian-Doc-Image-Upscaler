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

COPY = {
    "fa": {
        "brand": "دقیق‌خوان",
        "tagline": "بهبود تصویر و استخراج دقیق متن فارسی",
        "ready": "موتور OCR آماده است",
        "upload_title": "تصویر ورودی",
        "upload_sub": "یک تصویر یا اسکن دارای متن فارسی انتخاب کنید.",
        "file": "انتخاب تصویر",
        "profile": "نوع تصویر",
        "scale": "ضریب افزایش ابعاد",
        "action": "بهبود تصویر و استخراج متن",
        "enhanced": "تصویر بهبودیافته",
        "ocr_view": "نمای آماده‌شده برای OCR",
        "text": "متن استخراج‌شده",
        "txt": "دریافت فایل متن",
        "hint": "برای متن چاپی «سند» بهترین نقطه شروع است.",
    },
    "en": {
        "brand": "DaqiqKhan",
        "tagline": "Persian image enhancement and accurate OCR",
        "ready": "OCR engine ready",
        "upload_title": "Input image",
        "upload_sub": "Choose an image or scan containing Persian text.",
        "file": "Select image",
        "profile": "Image profile",
        "scale": "Upscale factor",
        "action": "Enhance image and extract text",
        "enhanced": "Enhanced image",
        "ocr_view": "OCR preprocessing view",
        "text": "Extracted text",
        "txt": "Download text file",
        "hint": "Document mode is the best starting point for printed Persian text.",
    },
}


def header_html(language: str) -> str:
    t = COPY[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='header-inner' dir='{direction}'>"
        "<div class='brand'><div class='brand-logo'>د</div><div>"
        f"<div class='brand-name'>{t['brand']}</div>"
        f"<div class='brand-tagline'>{t['tagline']}</div></div></div>"
        f"<div class='ready'><i></i>{t['ready']}</div>"
        "</div>"
    )


def intro_html(language: str) -> str:
    t = COPY[language]
    direction = "ltr" if language == "en" else "rtl"
    return (
        f"<div class='intro' dir='{direction}'>"
        f"<strong>{t['upload_title']}</strong><span>{t['upload_sub']}</span></div>"
    )


def hint_html(language: str) -> str:
    direction = "ltr" if language == "en" else "rtl"
    return f"<div class='hint' dir='{direction}'>{COPY[language]['hint']}</div>"


def localize(language: str):
    t = COPY[language]
    text_class = ["ltr"] if language == "en" else ["rtl"]
    return (
        gr.HTML(value=header_html(language)),
        gr.HTML(value=intro_html(language)),
        gr.File(label=t["file"]),
        gr.Radio(choices=PROFILE_CHOICES[language], label=t["profile"]),
        gr.Slider(label=t["scale"]),
        gr.Button(value=t["action"]),
        gr.HTML(value=hint_html(language)),
        gr.Image(label=t["enhanced"]),
        gr.Image(label=t["ocr_view"]),
        gr.Textbox(label=t["text"], elem_classes=text_class),
        gr.File(label=t["txt"]),
    )


def run(file_obj, profile, scale, language):
    if file_obj is None:
        message = (
            "Please select an image containing Persian text."
            if language == "en"
            else "لطفاً یک تصویر دارای متن فارسی انتخاب کنید."
        )
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

html, body {height:100%; margin:0; overflow:hidden !important;}
body, .gradio-container {font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif !important;}
.gradio-container {
  max-width:1540px !important;
  width:100% !important;
  height:100dvh !important;
  margin:0 auto !important;
  padding:12px 16px !important;
  overflow:hidden !important;
  box-sizing:border-box !important;
}
footer, .footer, .built-with {display:none !important;}

#header {height:58px !important; margin:0 0 10px !important;}
.header-inner {height:58px; display:flex; align-items:center; justify-content:space-between; gap:16px;}
.brand {display:flex; align-items:center; gap:11px; min-width:0;}
.brand-logo {width:40px; height:40px; border-radius:13px; display:grid; place-items:center; color:#fff; font-size:20px; font-weight:800; background:linear-gradient(145deg,#5b5cf0,#7c3aed); box-shadow:0 10px 30px rgba(91,92,240,.25);}
.brand-name {font-size:1.08rem; font-weight:800; line-height:1.2;}
.brand-tagline {font-size:.72rem; opacity:.52; margin-top:3px;}
.ready {display:flex; align-items:center; gap:7px; padding:7px 11px; border-radius:999px; border:1px solid rgba(34,197,94,.22); background:rgba(34,197,94,.07); font-size:.72rem; white-space:nowrap;}
.ready i {width:7px; height:7px; border-radius:50%; background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,.12);}

#workspace {display:flex !important; direction:rtl; gap:12px !important; height:calc(100dvh - 92px) !important; min-height:0 !important;}
#controls {flex:0 0 31% !important; max-width:31% !important; min-width:320px !important;}
#results {flex:1 1 auto !important; min-width:0 !important;}
.panel {height:100% !important; min-height:0 !important; overflow:hidden !important; border:1px solid var(--border-color-primary) !important; background:var(--background-fill-secondary) !important; border-radius:18px !important; padding:14px !important; box-sizing:border-box !important;}

#controls {display:flex !important; flex-direction:column !important; gap:8px !important;}
#utility {gap:8px !important; margin:0 !important;}
#language {flex:1 1 auto !important; min-width:0 !important;}
#theme-toggle {flex:0 0 46px !important; width:46px !important; min-width:46px !important;}
#theme-toggle button {height:38px !important; min-height:38px !important; border-radius:11px !important;}
.intro {margin:2px 0 3px;}
.intro strong {display:block; font-size:.95rem; font-weight:800;}
.intro span {display:block; font-size:.72rem; opacity:.5; margin-top:2px; line-height:1.5;}
#upload {max-height:150px !important; min-height:120px !important; overflow:hidden !important;}
#upload > div {min-height:110px !important;}
#profile, #scale {margin:0 !important;}
#action button {height:46px !important; min-height:46px !important; border-radius:12px !important; font-weight:800 !important; font-size:.9rem !important;}
.hint {font-size:.7rem; opacity:.54; line-height:1.7;}

#results {display:flex !important; flex-direction:column !important; gap:10px !important;}
#image-row {display:flex !important; flex:0 0 57% !important; min-height:0 !important; gap:10px !important;}
.preview {flex:1 1 50% !important; min-width:0 !important; height:100% !important;}
.preview > div {height:100% !important; min-height:0 !important;}
.preview img {object-fit:contain !important;}
#text-row {display:flex !important; flex:1 1 43% !important; min-height:0 !important; gap:10px !important;}
#ocr-text {flex:1 1 auto !important; min-width:0 !important; height:100% !important;}
#ocr-text textarea {height:calc(100% - 34px) !important; min-height:150px !important; resize:none !important;}
#txt-file {flex:0 0 190px !important; max-width:190px !important; align-self:flex-end !important;}
.rtl, .rtl textarea, .rtl input {direction:rtl !important; text-align:right !important;}
.ltr, .ltr textarea, .ltr input {direction:ltr !important; text-align:left !important; font-family:'Inter',sans-serif !important;}

body.light-mode {color-scheme:light; background:#f5f7fb !important;}
body.light-mode .gradio-container {background:#f5f7fb !important; color:#101828 !important;}
body.light-mode .panel {background:#fff !important; border-color:#e4e7ec !important;}
body.light-mode .ready {color:#157f3b; background:#edfdf2; border-color:#b7ebc6;}

@media (max-width:980px) {
  html, body {overflow:auto !important;}
  .gradio-container {height:auto !important; min-height:100dvh !important; overflow:visible !important; padding:10px !important;}
  #workspace {height:auto !important; flex-wrap:wrap !important;}
  #controls, #results {flex:1 1 100% !important; max-width:100% !important; min-width:0 !important; height:auto !important; overflow:visible !important;}
  #image-row {height:480px !important; flex-basis:480px !important;}
  #text-row {min-height:320px !important;}
}
"""

INIT_JS = """
() => {
  const theme = localStorage.getItem('dq-theme') || 'dark';
  document.body.classList.toggle('light-mode', theme === 'light');
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
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'en' ? 'ltr' : 'rtl';
  return lang;
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan") as demo:
    header = gr.HTML(header_html("fa"), elem_id="header")

    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_id="controls", elem_classes=["panel"]):
            with gr.Row(elem_id="utility"):
                language = gr.Radio(
                    [("فارسی", "fa"), ("English", "en")],
                    value="fa",
                    label=None,
                    elem_id="language",
                )
                theme = gr.Button("☼", elem_id="theme-toggle")

            intro = gr.HTML(intro_html("fa"))
            input_file = gr.File(
                label=COPY["fa"]["file"],
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
                elem_id="upload",
            )
            profile = gr.Radio(
                PROFILE_CHOICES["fa"],
                value="سند",
                label=COPY["fa"]["profile"],
                elem_id="profile",
            )
            scale = gr.Slider(
                1.0,
                3.0,
                value=2.0,
                step=0.5,
                label=COPY["fa"]["scale"],
                elem_id="scale",
            )
            submit = gr.Button(
                COPY["fa"]["action"],
                variant="primary",
                elem_id="action",
            )
            hint = gr.HTML(hint_html("fa"))

        with gr.Column(elem_id="results", elem_classes=["panel"]):
            with gr.Row(elem_id="image-row"):
                output_image = gr.Image(
                    type="filepath",
                    label=COPY["fa"]["enhanced"],
                    elem_classes=["preview"],
                )
                ocr_preview = gr.Image(
                    type="filepath",
                    label=COPY["fa"]["ocr_view"],
                    elem_classes=["preview"],
                )

            with gr.Row(elem_id="text-row"):
                output_text = gr.Textbox(
                    lines=8,
                    label=COPY["fa"]["text"],
                    elem_id="ocr-text",
                    elem_classes=["rtl"],
                )
                text_file = gr.File(
                    label=COPY["fa"]["txt"],
                    elem_id="txt-file",
                )

    language.change(
        fn=localize,
        inputs=[language],
        outputs=[
            header,
            intro,
            input_file,
            profile,
            scale,
            submit,
            hint,
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
