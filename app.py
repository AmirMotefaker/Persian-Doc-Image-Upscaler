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
        raise gr.Error("لطفاً یک فایل تصویر انتخاب کنید.")
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    if not path:
        raise gr.Error("فایل ورودی معتبر نیست.")
    try:
        return process_image(path, profile, float(scale))
    except Exception as exc:
        raise gr.Error(f"پردازش تصویر ناموفق بود: {exc}") from exc


CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.hero {text-align:right; direction:rtl; padding:8px 0 14px;}
.hero h1 {font-size:2rem; margin-bottom:.35rem;}
.hero p {opacity:.82; font-size:1.02rem;}
.rtl {direction:rtl; text-align:right;}
"""

with gr.Blocks(title="بهبود تصویر و OCR فارسی", css=CSS) as demo:
    gr.HTML(
        "<div class='hero'><h1>بهبود کیفیت تصویر و OCR فارسی</h1>"
        "<p>برای اسناد و تصاویر دارای متن فارسی؛ با حفظ مسیر بصری جدا از تصویر بهینه‌شده برای OCR.</p></div>"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="فایل تصویر",
                file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"],
                type="filepath",
            )
            profile = gr.Radio(list(PROFILES), value="سند", label="پروفایل پردازش")
            scale = gr.Slider(1.0, 3.0, value=2.0, step=0.5, label="بزرگ‌نمایی غیرمولد")
            submit = gr.Button("افزایش کیفیت و استخراج متن", variant="primary")
            gr.Markdown(
                "**فرمت‌ها:** PNG، JPG/JPEG، WebP، BMP، TIFF  \n"
                "پردازش P0 غیرمولد است تا شکل حروف فارسی بدون hallucination مدل‌های SR حفظ شود.",
                elem_classes=["rtl"],
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("تصویر بهبودیافته"):
                    output_image = gr.Image(type="filepath", label="خروجی بصری")
                with gr.Tab("تصویر OCR"):
                    ocr_preview = gr.Image(type="filepath", label="پیش‌پردازش مخصوص OCR")
                with gr.Tab("متن فارسی"):
                    output_text = gr.Textbox(lines=16, label="OCR", show_copy_button=True, rtl=True)
                    text_file = gr.File(label="فایل TXT خروجی")

    submit.click(
        fn=run,
        inputs=[input_file, profile, scale],
        outputs=[output_image, ocr_preview, output_text, text_file],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
