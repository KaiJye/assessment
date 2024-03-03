# How to setup

1. Install poppler for Windows:

    https://github.com/oschwartz10612/poppler-windows/releases

2. Install Tesseract for Windows

    https://github.com/UB-Mannheim/tesseract/wiki

3. Clone this repository

4. Duplicate and rename ".env.example" to ".env"

5. Change "POPPLER_PATH" and "TESSERACT_PATH" of .env to appropriate paths

6. In a terminal, create a python virtual environment:

        python -m venv env

7. Activate the virtual environment:
        
        .\env\Scripts\Activate.ps1

8. Install dependencies:

        pip install -r requirements.txt

9. Start Streamlit application:

        streamlit run app.py
