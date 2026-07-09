import os
from PyPDF2 import PdfReader

def save_file_and_load_text(file_path: str) -> str:
    filename = os.path.basename(file_path)
    dest_path = os.path.join("docs", filename)

    if os.path.abspath(file_path) != os.path.abspath(dest_path):
        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())

    text = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif filename.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")

    return text
