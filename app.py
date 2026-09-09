from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gradio as gr

from persian_upscaler.enhancement import PROFILES
from persian_upscaler.service import process_image


def run(file_obj, profile, scale, language):
    is_en = language == "English"
    if file_obj is None:
        raise gr.Error("Please select an image containing Persian text." if is_en else "لطفاً یک تصویر دارای متن فارسی انتخاب کنید.")
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        raise gr.Error("Invalid input file." if is_en else "فایل ورودی معتبر نیست.")
    try:
        return process_image(path, profile, float(scale), "en" if is_en else "fa")
    except Exception as exc:
        prefix = "Image processing failed" if is_en else "پردازش تصویر ناموفق بود"
        raise gr.Error(f"{prefix}: {exc}") from exc


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;600;700;800&display=swap');

:root {color-scheme: dark;}
html, body {height:100%; margin:0; overflow:hidden !important;}
body, .gradio-container {font-family:'Vazirmatn','Inter',Tahoma,Arial,sans-serif !important;}
.gradio-container {
  max-width: 1480px !important;
  height:100vh !important;
  margin:0 auto !important;
  padding:14px 18px 12px !important;
  overflow:hidden !important;
  direction:rtl;
}
footer, .footer {display:none !important;}
.app-topbar {display:flex; align-items:center; justify-content:space-between; gap:16px; height:56px; margin-bottom:10px; direction:rtl;}
.brand-wrap {display:flex; align-items:center; gap:11px; min-width:0;}
.brand-mark {width:40px; height:40px; border-radius:13px; display:grid; place-items:center; font-size:21px; font-weight:800; background:linear-gradient(145deg,#5b5cf0,#8b5cf6); box-shadow:0 10px 28px rgba(91,92,240,.24);}
.brand-title {font-size:1.08rem; font-weight:800; line-height:1.2;}
.brand-sub {font-family:'Inter',sans-serif; opacity:.52; font-size:.72rem; margin-top:2px; direction:ltr; text-align:right;}
.top-note {font-size:.79rem; opacity:.62; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.workspace {gap:14px !important; height:calc(100vh - 82px) !important; align-items:stretch !important;}
.panel {height:100% !important; min-height:0 !important; overflow:hidden !important; border:1px solid rgba(255,255,255,.08) !important; background:rgba(255,255,255,.025) !important; border-radius:18px !important; padding:14px !important;}
.controls-panel {display:flex !important; flex-direction:column !important;}
.output-panel {display:flex !important; flex-direction:column !important;}
.section-head {direction:rtl; text-align:right; margin-bottom:8px;}
.section-head strong {display:block; font-size:.95rem; font-weight:800;}
.section-head span {display:block; opacity:.48; font-size:.72rem; margin-top:2px;}
.compact-row {gap:8px !important;}
.primary-action button, button.primary {min-height:44px !important; border-radius:12px !important; font-weight:800 !important; font-size:.9rem !important;}
.rtl, .rtl textarea, .rtl input {direction:rtl !important; text-align:right !important;}
.ltr, .ltr textarea, .ltr input {direction:ltr !important; text-align:left !important;}
.output-tabs {height:100% !important; min-height:0 !important;}
.output-tabs > div {height:100% !important; min-height:0 !important;}
.output-image {height:calc(100vh - 205px) !important; min-height:300px !important;}
.output-text textarea {height:calc(100vh - 315px) !important; min-height:240px !important;}
.hint {font-size:.72rem; opacity:.57; line-height:1.8; margin-top:4px;}
.status-chip {display:inline-flex; align-items:center; gap:6px; border:1px solid rgba(34,197,94,.22); background:rgba(34,197,94,.07); color:#b7f7c9; padding:6px 10px; border-radius:999px; font-size:.72rem; white-space:nowrap;}
.gradio-container.light-mode {filter:none;}
html.light-mode, body.light-mode {color-scheme:light; background:#f6f7fb !important;}
body.light-mode .gradio-container {background:#f6f7fb !important; color:#101828 !important;}
body.light-mode .panel {background:#fff !important; border-color:#e4e7ec !important;}
body.light-mode .status-chip {color:#157f3b; background:#edfdf2; border-color:#b7ebc6;}
body.light-mode .brand-sub, body.light-mode .top-note, body.light-mode .section-head span, body.light-mode .hint {opacity:.68;}
@media (max-width: 900px) {
  html, body {overflow:auto !important;}
  .gradio-container {height:auto !important; overflow:visible !important; padding:12px !important;}
  .workspace {height:auto !important;}
  .panel {height:auto !important; overflow:visible !important;}
  .output-image {height:420px !important;}
}
"""

JS = """
() => {
  const applyTheme = (light) => {
    document.documentElement.classList.toggle('light-mode', light);
    document.body.classList.toggle('light-mode', light);
    localStorage.setItem('dq-theme', light ? 'light' : 'dark');
  };
  applyTheme(localStorage.getItem('dq-theme') === 'light');
  const timer = setInterval(() => {
    const btn = document.querySelector('#theme-toggle button');
    if (btn && !btn.dataset.bound) {
      btn.dataset.bound = '1';
      btn.addEventListener('click', () => {
        applyTheme(!document.body.classList.contains('light-mode'));
      });
      clearInterval(timer);
    }
  }, 200);
}
"""

with gr.Blocks(title="دقیق‌خوان | DaqiqKhan Persian OCR") as demo:
    gr.HTML(
        "<div class='app-topbar'>"
        "<div class='brand-wrap'><div class='brand-mark'>د</div><div>"
        "<div class='brand-title'>دقیق‌خوان</div>"
        "<div class='brand-sub'>Persian Image Enhancement & OCR</div></div></div>"
        "<div class='top-note'>متن فارسی را واضح‌تر کنید و دقیق‌تر استخراج کنید · Enhance Persian text and extract it accurately</div>"
        "<div class='status-chip'>● OCR engine ready</div>"
        "</div>"
    )

    with gr.Row(elem_classes=["workspace"]):
        with gr.Column(scale=4, elem_classes=["panel", "controls-panel"]):
            with gr.Row(elem_classes=["compact-row"]):
                language = gr.Radio(["فارسی", "English"], value="فارسی", label="زبان / Language", scale=3)
                theme = gr.Button("◐", elem_id="theme-toggle", scale=1)

            gr.HTML("<div class='section-head'><strong>۱. تصویر / Image</strong><span>فایل دارای متن فارسی را انتخاب کنید · Select an image containing Persian text</span></div>")
            input_file = gr.File(
                label="انتخاب تصویر / Select image",
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
            )

            gr.HTML("<div class='section-head' style='margin-top:6px'><strong>۲. تنظیمات / Processing</strong><span>برای اسناد چاپی حالت «سند» مناسب است · Document mode is recommended for printed text</span></div>")
            profile = gr.Radio(list(PROFILES), value="سند", label="نوع تصویر / Profile")
            scale = gr.Slider(1.0, 3.0, value=2.0, step=0.5, label="افزایش ابعاد / Scale")
            submit = gr.Button("پردازش و استخراج متن / Process & OCR", variant="primary", elem_classes=["primary-action"])
            gr.HTML("<div class='hint'>طبیعی: عکس واضح · سند: متن چاپی · اسکن ضعیف: تصویر کم‌کیفیت<br>Natural: clean photo · Document: printed text · Weak scan: low-quality scan</div>")

        with gr.Column(scale=8, elem_classes=["panel", "output-panel"]):
            gr.HTML("<div class='section-head'><strong>۳. نتیجه / Results</strong><span>خروجی تصویر، نمای OCR و متن استخراج‌شده · Enhanced image, OCR view and extracted text</span></div>")
            with gr.Tabs(elem_classes=["output-tabs"]):
                with gr.Tab("تصویر بهبودیافته / Enhanced"):
                    output_image = gr.Image(type="filepath", label=None, elem_classes=["output-image"])
                with gr.Tab("نمای OCR / OCR view"):
                    ocr_preview = gr.Image(type="filepath", label=None, elem_classes=["output-image"])
                with gr.Tab("متن / Text"):
                    output_text = gr.Textbox(
                        lines=12,
                        label="متن استخراج‌شده / Extracted text",
                        elem_classes=["rtl", "output-text"],
                    )
                    text_file = gr.File(label="فایل متن / TXT output")

    submit.click(
        fn=run,
        inputs=[input_file, profile, scale, language],
        outputs=[output_image, ocr_preview, output_text, text_file],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS, js=JS)
