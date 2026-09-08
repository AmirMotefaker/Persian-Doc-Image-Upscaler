# 🇮🇷 Persian Doc & Image Upscaler

A free, open-source platform specifically designed to enhance the quality of Persian documents, images, and PDFs. It solves the common issue of blurred Persian characters (dots and teeth) in generic upscalers, coupled with accurate Persian OCR.

## ✨ Features
- **4x Super-Resolution**: Powered by Real-ESRGAN.
- **PDF Support**: Process multi-page Persian PDFs, enhance quality, and extract text.
- **Persian-Specific Post-Processing**: Custom OpenCV Adaptive Thresholding to preserve Persian diacritics (نقطه‌ها) and cursive connections (دندانه‌ها).
- **Built-in Persian OCR**: Integrated PaddleOCR optimized for the Persian language (`lang='fa'`).
- **100% Free & Open Source**.

## 🛠️ Tech Stack
- **Frontend**: Gradio
- **Upscaling**: Real-ESRGAN (BasicSR)
- **Image Processing**: OpenCV (Custom Persian text enhancement pipeline)
- **PDF Processing**: PyMuPDF (fitz)
- **OCR**: PaddleOCR

## 💻 Local Installation
```bash
git clone https://github.com/AmirMotefaker/Persian-Doc-Image-Upscaler.git
cd Persian-Doc-Image-Upscaler
pip install -r requirements.txt
python app.py