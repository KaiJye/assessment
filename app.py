from pdf2jpg import pdf2jpg
import yaml
from PIL import Image
from transformers import pipeline
import pytesseract
import pandas as pd
import cv2 as cv
import numpy as np
import locale
import datetime
import streamlit as st


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MODEL_PATH = r"microsoft/trocr-base-printed"
TEMPLATES_YAML_PATH = r"templates.yaml"
TM_PDF_JPG_TEMP_PATH = r"TM.pdf_dir/0_TM.pdf.jpg"
TM_PDF_PATH = r"TM.pdf"
TNB_PHYSICAL_PATH = r"tnb_physical.jpg"
TNB_DIGITAL_PATH = r"tnb_digital.jpg"
TM_PDF_TEMPLATE_KEY = r"tm"
TNB_PHYSICAL_TEMPLATE_KEY = r"tnb_physical"
TNB_DIGITAL_TEMPLATE_KEY = r"tnb_digital"
ADDRESS_KEY = r"address"
DATE_KEY = r"date"
NAME_KEY = r"name"
BOUNDING_BOX_PERCENT_KEY = r"bounding_box_percent"


def get_ocr(model_path):
    ocr = pipeline(model=model_path, task="image-to-text")

    return ocr


def infer(img, ocr):
    if not isinstance(img, Image.Image):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    generated_text = ocr(img, max_new_tokens=20)

    return generated_text[0]["generated_text"]


def load_image(path):
    return cv.imread(path)


def load_templates_yaml(path):
    with open(path, 'r') as file:
        templates = yaml.safe_load(file)

    return templates


def convert_and_save_tm_pdf_to_jpg(pdf_path):
    pdf2jpg.convert_pdf2jpg(pdf_path, '.', dpi=200, pages="0")


def add_margin(img: cv.Mat, margin):
    img = cv.copyMakeBorder(img, margin, margin, margin, margin, cv.BORDER_CONSTANT, value=(255, 255, 255))
    return img


def crop_percentage(img: cv.Mat, x1, x2, y1, y2):
    height, width, _ = img.shape

    left = int(width * x1 / 100)
    right = int(width * x2 / 100)
    top = int(height * y1 / 100)
    bottom = int(height * y2 / 100)

    return crop(img, left, right, top, bottom)


def crop(img: cv.Mat, x1, x2, y1, y2):
    return img[y1:y2, x1:x2]


def get_bounding_box_percents(templates, template_key, field_key):
    rect = templates[template_key][field_key][BOUNDING_BOX_PERCENT_KEY]
    return rect


def read_multiline(img, ocr, templates, template_key, field_key) -> str:
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
    rect = get_bounding_box_percents(templates, template_key, field_key)
    img = crop_percentage(img, **rect)

    text = infer(img, ocr)

    return text


def process_tm_pdf(ocr, templates):
    convert_and_save_tm_pdf_to_jpg(TM_PDF_PATH)
    img = load_image(TM_PDF_JPG_TEMP_PATH)

    name_text = read_single_line(img, ocr, templates, TM_PDF_TEMPLATE_KEY, NAME_KEY)
    print(f"name: {name_text}")

    date_text = read_single_line(img, ocr, templates, TM_PDF_TEMPLATE_KEY, DATE_KEY)
    date_text = date_text.strip().replace(' ', '')
    locale.setlocale(locale.LC_TIME, 'en')
    date_text = datetime.datetime.strptime(date_text, "%d%b%Y").strftime("%y%m%d")
    print(f"date: {date_text}")

    address_text = read_multiline(img, ocr, templates, TM_PDF_TEMPLATE_KEY, ADDRESS_KEY)
    print(f"address: {address_text}")


def process_tnb_digital(ocr, templates):
    img = load_image(TNB_DIGITAL_PATH)

    name_text = read_single_line(img, ocr, templates, TNB_DIGITAL_TEMPLATE_KEY, NAME_KEY)
    print(f"name: {name_text}")

    date_text = read_single_line(img, ocr, templates, TNB_DIGITAL_TEMPLATE_KEY, DATE_KEY)
    date_text = date_text.strip().replace(' ', '')
    locale.setlocale(locale.LC_TIME, 'ms')
    date_text = datetime.datetime.strptime(date_text, "%d.%m.%Y").strftime("%y%m%d")
    print(f"date: {date_text}")

    address_text = read_multiline(img, ocr, templates, TNB_DIGITAL_TEMPLATE_KEY, ADDRESS_KEY)
    print(f"address: {address_text}")


def process_tnb_physical(ocr, templates):
    img = load_image(TNB_PHYSICAL_PATH)

    name_text = read_single_line(img, ocr, templates, TNB_PHYSICAL_TEMPLATE_KEY, NAME_KEY)
    print(f"name: {name_text}")

    st.write(name_text)

    # date_text = read_single_line(img, ocr, templates, TNB_PHYSICAL_TEMPLATE_KEY, DATE_KEY)
    # date_text = date_text.strip().replace(' ', '')
    # locale.setlocale(locale.LC_TIME, 'ms')
    # date_text = datetime.datetime.strptime(date_text, "%d%b%Y").strftime("%y%m%d")
    # print(f"date: {date_text}")

    # address_text = read_multiline(img, ocr, templates, TNB_PHYSICAL_TEMPLATE_KEY, ADDRESS_KEY)
    # print(f"address: {address_text}")


def main():
    locale.setlocale(locale.LC_TIME, 'ms_MY')
    ocr = None
    ocr = get_ocr(MODEL_PATH)
    templates = load_templates_yaml(TEMPLATES_YAML_PATH)

    # process_tm_pdf(ocr, templates)
    # process_tnb_digital(ocr, templates)
    process_tnb_physical(ocr, templates)


if __name__ == "__main__":
    main()
    # print(datetime.datetime.strptime("01JUL2019", "%d%b%Y"))

