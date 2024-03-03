import os
import yaml
import json
from PIL import Image
from transformers import pipeline
import pytesseract
import pandas as pd
import cv2 as cv
import numpy as np
import locale
import datetime
import streamlit as st
from pdf2image import convert_from_bytes, convert_from_path
from dotenv import load_dotenv

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")
POPPLER_PATH = os.getenv("POPPLER_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
TEMPLATES_YAML_PATH = r"templates.yaml"
TM_PDF_TEMPLATE_KEY = r"tm"
TNB_PHYSICAL_TEMPLATE_KEY = r"tnb_physical"
TNB_DIGITAL_TEMPLATE_KEY = r"tnb_digital"
ADDRESS_KEY = r"address"
DATE_KEY = r"date"
NAME_KEY = r"name"
BOUNDING_BOX_PERCENT_KEY = r"bounding_box_percent"


def get_ocr(model_path):
    """Initialize OCR model
    model_path: 'microsoft/trocr-base-printed' or 'microsoft/trocr-small-printed'
    """
    ocr = pipeline(model=model_path, task="image-to-text")

    return ocr


def infer(img, ocr):
    """OCR inference
    img: opencv Mat, must contain single line text
    ocr: OCR model
    """
    if not isinstance(img, Image.Image):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    generated_text = ocr(img, max_new_tokens=20)

    return generated_text[0]["generated_text"]


def load_templates_yaml(path):
    """Load template object
    path: path to template.yaml
    """
    with open(path, 'r') as file:
        templates = yaml.safe_load(file)

    return templates


def convert_tm_pdf_to_img(data):
    """Convert TM.pdf to opencv Mat object
    data: pdf file in bytes
    """
    images = convert_from_bytes(
        data, first_page=0, last_page=1, 
        poppler_path=POPPLER_PATH
    )

    img = np.array(images[0])
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    return img


def add_margin(img: cv.Mat, margin):
    """Add white margin to all sides
    img: opencv Mat
    margin: margin width (int)"""
    img = cv.copyMakeBorder(img, margin, margin, margin, margin, cv.BORDER_CONSTANT, value=(255, 255, 255))
    return img


def crop_percentage(img: cv.Mat, x1, x2, y1, y2):
    """Crop image by percentage
    img: opencv Mat
    x1: left (0.0 - 1.0)
    x2: right (0.0 - 1.0)
    y1: top (0.0 - 1.0)
    y2: bottom (0.0 - 1.0)"""
    height, width, _ = img.shape

    left = int(width * x1 / 100)
    right = int(width * x2 / 100)
    top = int(height * y1 / 100)
    bottom = int(height * y2 / 100)

    return crop(img, left, right, top, bottom)


def crop(img: cv.Mat, x1, x2, y1, y2):
    """Crop image by pixel
    img: opencv Mat
    x1: left (int)
    x2: right (int)
    y1: top (int)
    y2: bottom (int)"""
    return img[y1:y2, x1:x2]


def get_bounding_box_percents(templates, template_key, field_key):
    """get bounding box (percentage) from template
    templates: template object of templates.yaml
    template_key: 'tm' or 'tnb_digital' or 'tnb_physical'
    field_key: 'address' or 'date' or 'name'
    """
    rect = templates[template_key][field_key][BOUNDING_BOX_PERCENT_KEY]
    return rect


def read_multiline(img, ocr, templates, template_key, field_key) -> str:
    """Read image containing multiple lines of text
    img: opencv Mat
    ocr: OCR model
    templates: template object of templates.yaml
    template_key: 'tm' or 'tnb_digital' or 'tnb_physical'
    field_key: 'address' or 'date' or 'name'
    """
    rect = get_bounding_box_percents(templates, template_key, field_key)
    img = crop_percentage(img, **rect)

    ocr_img = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    img = clahe.apply(img)

    img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,33,25)

    img = add_margin(img, 10)
    ocr_img = add_margin(ocr_img, 10)

    data: pd.DataFrame = pytesseract.image_to_data(img, output_type='data.frame')
    data = data[data.level == 4]

    texts = []
    for i, row in data.iterrows():
        top, left, width, height = row["top"], row["left"], row["width"], row["height"]
        line_image = crop(ocr_img, left, left+width, top, top+height)

        text = infer(line_image, ocr)
        texts.append(text)

    return "\n".join(texts)


def read_single_line(img, ocr, templates, template_key, field_key) -> str:
    """Read image containing a single line of text
    img: opencv Mat
    ocr: OCR model
    templates: template object of templates.yaml
    template_key: 'tm' or 'tnb_digital' or 'tnb_physical'
    field_key: 'address' or 'date' or 'name'
    """
    rect = get_bounding_box_percents(templates, template_key, field_key)
    img = crop_percentage(img, **rect)

    text = infer(img, ocr)

    return text


def to_json(address, date, name):
    """Format address, date and name into JSON
    """
    return json.dumps(
        {
            "address": address,
            "date": date,
            "name": name,
        }
    )


def convert_date(date_text, original_format, locale_str):
    """Convert date into "%y%m%d" format. (example 230321)
    date_text: date to be converted
    original_format: orignal format of date_text
    locale_str: locale of date_text, 'ms' for Malay, 'en' for English
    """
    date_text = date_text.strip().replace(' ', '')
    original_format = original_format.strip().replace(' ', '')
    locale.setlocale(locale.LC_TIME, locale_str)
    date_text = datetime.datetime.strptime(date_text, original_format).strftime("%y%m%d")
    return date_text


def process_tm_pdf(ocr, templates):
    """Draw file upload element to process TM.pdf
    ocr: OCR model
    templates: templates.yaml object
    """
    with st.form("TM.pdf_form"):
        uploaded_file = st.file_uploader("Upload TM.pdf")

        submitted = st.form_submit_button("Submit")
        
        if submitted:

            if uploaded_file is None:
                st.info("No file uploaded")

                return
            
            with st.spinner("Reading file ..."):
            
                img = convert_tm_pdf_to_img(uploaded_file.getvalue())

                name_text = read_single_line(img, ocr, templates, TM_PDF_TEMPLATE_KEY, NAME_KEY)
                print(f"name: {name_text}")

                date_text = read_single_line(img, ocr, templates, TM_PDF_TEMPLATE_KEY, DATE_KEY)
                date_text = convert_date(date_text, "%d%b%Y", 'en')
                print(f"date: {date_text}")

                address_text = read_multiline(img, ocr, templates, TM_PDF_TEMPLATE_KEY, ADDRESS_KEY)
                print(f"address: {address_text}")

            st.json(to_json(address_text, date_text, name_text))


def process_tnb_digital(ocr, templates):
    """Draw file upload element to process tnb_digital.jpg
    ocr: OCR model
    templates: templates.yaml object
    """
    with st.form("tnb_digital.jpg_form"):
        uploaded_file = st.file_uploader("Upload tnb_digital.jpg")

        submitted = st.form_submit_button("Submit")
        
        if submitted:

            if uploaded_file is None:
                st.info("No file uploaded")

                return
            
            with st.spinner("Reading file ..."):
                img = np.frombuffer(uploaded_file.getvalue(), np.uint8)
                img = cv.imdecode(img, cv.IMREAD_COLOR)

                name_text = read_single_line(img, ocr, templates, TNB_DIGITAL_TEMPLATE_KEY, NAME_KEY)
                print(f"name: {name_text}")

                date_text = read_single_line(img, ocr, templates, TNB_DIGITAL_TEMPLATE_KEY, DATE_KEY)
                date_text = convert_date(date_text, "%d.%m.%Y", 'ms')
                print(f"date: {date_text}")

                address_text = read_multiline(img, ocr, templates, TNB_DIGITAL_TEMPLATE_KEY, ADDRESS_KEY)
                print(f"address: {address_text}")

            st.json(to_json(address_text, date_text, name_text))


def process_tnb_physical(ocr, templates):
    """Draw file upload element to process tnb_physical.jpg
    ocr: OCR model
    templates: templates.yaml object
    """
    with st.form("tnb_physical.jpg_form"):
        uploaded_file = st.file_uploader("Upload tnb_physical.jpg")

        submitted = st.form_submit_button("OCR")
        
        if submitted:
            if uploaded_file is None:
                st.info("No file uploaded")

                return
            
            with st.spinner("Reading file ..."):
                img = np.frombuffer(uploaded_file.getvalue(), np.uint8)
                img = cv.imdecode(img, cv.IMREAD_COLOR)

                name_text = read_single_line(img, ocr, templates, TNB_PHYSICAL_TEMPLATE_KEY, NAME_KEY)
                print(f"name: {name_text}")

                date_text = read_single_line(img, ocr, templates, TNB_PHYSICAL_TEMPLATE_KEY, DATE_KEY)
                date_text = convert_date(date_text, "%d%b%Y", 'ms')
                print(f"date: {date_text}")

                address_text = read_multiline(img, ocr, templates, TNB_PHYSICAL_TEMPLATE_KEY, ADDRESS_KEY)
                print(f"address: {address_text}")

            st.json(to_json(address_text, date_text, name_text))


def main():
    """Entry point
    """
    ocr = get_ocr(MODEL_PATH)
    templates = load_templates_yaml(TEMPLATES_YAML_PATH)

    process_tm_pdf(ocr, templates)
    process_tnb_digital(ocr, templates)
    process_tnb_physical(ocr, templates)


if __name__ == "__main__":
    main()


