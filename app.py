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


def run(file_obj, profile, scale):
    if file_obj is None:
        raise gr.Error("لطفاً یک تصویر دارای متن فارسی انتخاب کنید.")
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        raise gr.Error("فایل ورودی معتبر نیست.")
    try:
        return process_image(path, profile, float(scale))
    except Exception as exc:
        raise gr.Error(f"پردازش تصویر ناموفق بود: {exc}") from exc


CSS = """
:root {color-scheme: dark;}
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding: 26px 28px 44px !important;
  direction: rtl;
}
body, .gradio-container {font-family: Tahoma, Arial, sans-serif !important;}
.app-shell {direction:rtl; text-align:right; margin-bottom:22px;}
.brand-row {display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;}
.brand {display:flex; align-items:center; gap:13px;}
.brand-mark {width:48px; height:48px; border-radius:15px; display:grid; place-items:center; font-size:25px; background:linear-gradient(145deg,#5b5cf0,#8b5cf6); box-shadow:0 12px 35px rgba(91,92,240,.24);}
.brand-title {font-size:1.28rem; font-weight:800; letter-spacing:-.02em;}
.brand-sub {opacity:.58; font-size:.84rem; margin-top:3px;}
.status-pill {border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.045); padding:8px 13px; border-radius:999px; font-size:.82rem; opacity:.9;}
.hero-panel {margin-top:20px; border:1px solid rgba(255,255,255,.08); background:linear-gradient(135deg,rgba(91,92,240,.12),rgba(139,92,246,.035) 58%,rgba(255,255,255,.02)); border-radius:24px; padding:28px 30px;}
.eyebrow {font-size:.8rem; color:#aaa9ff; font-weight:700; margin-bottom:9px;}
.hero-panel h1 {font-size:2.05rem; line-height:1.5; margin:0; letter-spacing:-.035em;}
.hero-panel p {max-width:820px; margin:9px 0 0; opacity:.65; line-height:2; font-size:.96rem;}
.trust-row {display:flex; gap:9px; flex-wrap:wrap; margin-top:17px;}
.trust-chip {border:1px solid rgba(255,255,255,.08); background:rgba(0,0,0,.18); padding:7px 11px; border-radius:10px; font-size:.78rem; opacity:.8;}
.workspace {gap:18px !important; align-items:stretch !important;}
.panel {border:1px solid rgba(255,255,255,.08) !important; background:rgba(255,255,255,.025) !important; border-radius:20px !important; padding:18px !important;}
.section-title {direction:rtl; text-align:right; margin-bottom:12px;}
.section-title strong {display:block; font-size:1rem;}
.section-title span {display:block; opacity:.52; font-size:.78rem; margin-top:4px;}
.primary-action button, button.primary {min-height:48px !important; border-radius:13px !important; font-weight:800 !important; font-size:.95rem !important;}
.rtl, .rtl textarea, .rtl input {direction:rtl !important; text-align:right !important;}
.output-area {min-height:500px;}
.footer-note {direction:rtl; text-align:center; opacity:.46; font-size:.75rem; padding-top:18px;}
.tabs {border-radius:16px !important;}
@media (max-width: 800px) {
  .gradio-container {padding:16px 12px 30px !important;}
  .hero-panel {padding:21px 18px; border-radius:18px;}
  .hero-panel h1 {font-size:1.55rem;}
}
"""

with gr.Blocks(title="دقیق‌خوان | بهبود تصویر و OCR فارسی") as demo:
    gr.HTML(
        "<div class='app-shell'>"
        "<div class='brand-row'><div class='brand'>"
        "<div class='brand-mark'>ض</div><div><div class='brand-title'>دقیق‌خوان</div>"
        "<div class='brand-sub'>Persian Image Enhancement & OCR</div></div></div>"
        "<div class='status-pill'>● موتور OCR فارسی آماده است</div></div>"
        "<div class='hero-panel'><div class='eyebrow'>پردازش تخصصی متن فارسی</div>"
        "<h1>متن فارسی را واضح‌تر ببینید، دقیق‌تر استخراج کنید.</h1>"
        "<p>تصویر یا اسکن خود را وارد کنید. مسیر بهبود بصری و مسیر OCR جداگانه پردازش می‌شوند تا خوانایی بیشتر شود و هندسه حروف، نقطه‌ها و دندانه‌های فارسی تا حد ممکن حفظ شود.</p>"
        "<div class='trust-row'><span class='trust-chip'>PP-OCRv5 فارسی</span>"
        "<span class='trust-chip'>پردازش غیرمولد</span><span class='trust-chip'>PNG · JPEG · WebP · BMP · TIFF</span>"
        "<span class='trust-chip'>خروجی TXT</span></div></div></div>"
    )

    with gr.Row(elem_classes=["workspace"]):
        with gr.Column(scale=4, elem_classes=["panel"]):
            gr.HTML("<div class='section-title'><strong>۱. تصویر ورودی</strong><span>تصویر، اسکن یا عکس دارای متن فارسی را انتخاب کنید.</span></div>")
            input_file = gr.File(
                label="انتخاب تصویر",
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
            )
            gr.HTML("<div class='section-title' style='margin-top:10px'><strong>۲. تنظیم پردازش</strong><span>برای اسناد چاپی، حالت «سند» پیشنهاد می‌شود.</span></div>")
            profile = gr.Radio(list(PROFILES), value="سند", label="نوع تصویر")
            scale = gr.Slider(1.0, 3.0, value=2.0, step=0.5, label="ضریب افزایش ابعاد")
            submit = gr.Button("بهبود تصویر و استخراج متن فارسی", variant="primary", elem_classes=["primary-action"])
            gr.Markdown(
                "**راهنما:** «طبیعی» برای عکس‌های واضح، «سند» برای متن چاپی و «اسکن ضعیف» برای تصاویر کم‌کیفیت مناسب است.",
                elem_classes=["rtl"],
            )

        with gr.Column(scale=8, elem_classes=["panel", "output-area"]):
            gr.HTML("<div class='section-title'><strong>۳. نتیجه پردازش</strong><span>خروجی بصری، نمای مخصوص OCR و متن استخراج‌شده را مقایسه کنید.</span></div>")
            with gr.Tabs():
                with gr.Tab("تصویر بهبودیافته"):
                    output_image = gr.Image(type="filepath", label="خروجی نهایی تصویر", height=430)
                with gr.Tab("نمای مخصوص OCR"):
                    ocr_preview = gr.Image(type="filepath", label="تصویر آماده‌شده برای تشخیص متن", height=430)
                with gr.Tab("متن استخراج‌شده"):
                    output_text = gr.Textbox(
                        lines=17,
                        label="متن فارسی شناسایی‌شده و گزارش اطمینان",
                        elem_classes=["rtl"],
                    )
                    text_file = gr.File(label="دریافت فایل متنی")

    gr.HTML("<div class='footer-note'>نسخه آزمایشی P0 · پردازش محلی · تمرکز بر حفظ ساختار نوشتار فارسی</div>")

    submit.click(
        fn=run,
        inputs=[input_file, profile, scale],
        outputs=[output_image, ocr_preview, output_text, text_file],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS)
