import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from deep_translator import GoogleTranslator

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(uploaded_image):
    # Convert to NumPy array
    img_np = np.array(uploaded_image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoising
    denoised = cv2.medianBlur(thresh, 3)

    return denoised

def extract_text_and_translate(image_np):
    # OCR using Tesseract
    ocr_text = pytesseract.image_to_string(image_np)

    # Translate to English using Deep Translator
    if ocr_text.strip():
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(ocr_text)
        except Exception as e:
            translated = f"Translation failed: {e}"
    else:
        translated = " No readable text found in the image."

    return ocr_text, translated

def main():
    st.title(" ReScript: Ancient Text OCR & Translation")

    st.write("Upload a photo of an ancient or Latin inscription to extract and translate the text.")

    image_file = st.file_uploader(" Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.info(" Processing image...")
        processed_img = preprocess_image(image)

        st.image(processed_img, caption=" Preprocessed Image (for better OCR)", clamp=True)

        st.success(" Image processed. Running OCR...")
        ocr_text, translated_text = extract_text_and_translate(processed_img)

        st.subheader(" Raw OCR Output")
        st.code(ocr_text.strip() if ocr_text.strip() else "No text found.")

        st.subheader(" Translated English Text")
        st.success(translated_text)

    # Optional: Use local image for testing
    if st.button(" Test on local image 'sample_img/hii.png'"):
        try:
            local_image = Image.open("sample_img/hii.png")
            st.image(local_image, caption="Local Test Image")

            processed = preprocess_image(local_image)
            ocr_text, translated = extract_text_and_translate(processed)

            st.subheader(" Raw OCR Output")
            st.code(ocr_text.strip())

            st.subheader(" Translated English Text")
            st.success(translated)
        except Exception as e:
            st.error(f"Error loading local image: {e}")

if __name__ == "__main__":
    main()
