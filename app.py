import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
from paddleocr import PaddleOCR
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import fitz

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=model,
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=False
)

ocr_engine = PaddleOCR(use_angle_cls=True, lang='fa', use_gpu=False, show_log=False)

def enhance_persian_image(img_cv):
    output, _ = upsampler.enhance(img_cv, outscale=4)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def process_image_file(image):
    if image is None:
        return None, "لطفاً یک تصویر آپلود کنید."
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    enhanced_img = enhance_persian_image(img_cv)
    enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    
    temp_path = "temp_ocr.png"
    cv2.imwrite(temp_path, enhanced_img)
    
    try:
        result = ocr_engine.ocr(temp_path, cls=True)
        extracted_text = ""
        confidence = 0.0
        count = 0
        
        if result and result[0]:
            for line in result[0]:
                extracted_text += line[1][0] + "\n"
                confidence += line[1][1]
                count += 1
                
        avg_conf = (confidence / count * 100) if count > 0 else 0
        ocr_output = f"✅ متن استخراج شده:\n\n{extracted_text}\n\n📊 میانگین اطمینان: {avg_conf:.2f}%"
    except Exception as e:
        ocr_output = f"❌ خطا در OCR: {str(e)}"
    
    if os.path.exists(temp_path): os.remove(temp_path)
    return enhanced_img_rgb, ocr_output

def process_pdf_file(pdf_file):
    if pdf_file is None:
        return None, "لطفاً یک فایل PDF آپلود کنید."
    
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    num_pages = len(doc)
    enhanced_pages = []
    all_text = ""
    temp_dir = "temp_pdf_pages"
    os.makedirs(temp_dir, exist_ok=True)
    
    progress_text = f"⏳ در حال پردازش {num_pages} صفحه...\n"
    
    for page_num in range(num_pages):
        page = doc[page_num]
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img_cv = img_data.reshape((pix.height, pix.width, pix.n))
        
        if pix.n == 4:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            
        enhanced_page = enhance_persian_image(img_cv)
        enhanced_pages.append(enhanced_page)
        
        try:
            temp_page_path = os.path.join(temp_dir, f"page_{page_num}.png")
            cv2.imwrite(temp_page_path, enhanced_page)
            
            ocr_result = ocr_engine.ocr(temp_page_path, cls=True)
            page_text = f"\n--- صفحه {page_num + 1} ---\n"
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    page_text += line[1][0] + "\n"
            all_text += page_text
            
            if os.path.exists(temp_page_path): os.remove(temp_page_path)
        except Exception as e:
            all_text += f"\n--- صفحه {page_num + 1} ---\nخطا: {str(e)}\n"
            
        progress_text += f"✅ صفحه {page_num + 1} از {num_pages} پردازش شد\n"
    
    output_pdf_path = os.path.join(temp_dir, "enhanced_document.pdf")
    pdf_writer = fitz.open()
    
    for enhanced_page in enhanced_pages:
        enhanced_rgb = cv2.cvtColor(enhanced_page, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(enhanced_rgb)
        temp_img_path = os.path.join(temp_dir, "temp_page.png")
        img_pil.save(temp_img_path, "PNG")
        
        new_page = pdf_writer.new_page(width=img_pil.width, height=img_pil.height)
        new_page.insert_image(new_page.rect, filename=temp_img_path)
        if os.path.exists(temp_img_path): os.remove(temp_img_path)
        
    pdf_writer.save(output_pdf_path)
    doc.close()
    
    return output_pdf_path, f"{progress_text}\n\n📝 متن استخراج شده:\n{all_text}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Persian Doc Upscaler") as demo:
    gr.Markdown("# 🇮🇷 Persian Doc & Image Upscaler + OCR")
    gr.Markdown("افزایش کیفیت خودکار اسناد، تصاویر و PDFهای فارسی + استخراج متن با حفظ نقاط و دندانه‌ها")
    
    with gr.Tab("پردازش تصویر"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="تصویر را آپلود کنید (JPG, PNG, BMP, TIFF)")
                submit_img_btn = gr.Button("🚀 افزایش کیفیت و استخراج متن", variant="primary")
            with gr.Column():
                output_img = gr.Image(type="numpy", label="تصویر بهبودیافته (۴ برابر)")
                output_text_img = gr.Textbox(label="متن استخراج‌شده", lines=8, show_copy_button=True)
        submit_img_btn.click(fn=process_image_file, inputs=input_img, outputs=[output_img, output_text_img])
    
    with gr.Tab("پردازش PDF"):
        gr.Markdown("### پردازش اسناد PDF فارسی (چند صفحه‌ای)")
        with gr.Row():
            with gr.Column():
                input_pdf = gr.File(label="فایل PDF را آپلود کنید", file_types=[".pdf"])
                submit_pdf_btn = gr.Button("🚀 پردازش PDF و استخراج متن", variant="primary")
            with gr.Column():
                output_pdf = gr.File(label="PDF بهبودیافته (دانلود)")
                output_text_pdf = gr.Textbox(label="متن استخراج‌شده از تمام صفحات", lines=12, show_copy_button=True)
        submit_pdf_btn.click(fn=process_pdf_file, inputs=input_pdf, outputs=[output_pdf, output_text_pdf])

if __name__ == "__main__":
    demo.launch()
