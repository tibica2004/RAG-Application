import fitz
import os

def read_pdf(path: str) -> str:
    """Citește tot textul dintr-un PDF."""
    document = fitz.open(path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text

def get_pdf_file(folder:str):
    pdf_file=[]
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            pdf_file.append(os.path.join(folder,file))

    return pdf_file