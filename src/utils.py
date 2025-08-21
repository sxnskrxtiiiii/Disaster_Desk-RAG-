# src/utils.py
# PDF utilities using PyMuPDF for more reliable extraction
from typing import List, Tuple
import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def pdf_to_pages(path: str) -> List[Tuple[int, str]]:
    # Uses PyMuPDF (package name: pymupdf; import name: fitz)
    import fitz  # type: ignore
    doc = fitz.open(path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(doc):
        txt = page.get_text("text") or ""
        pages.append((i + 1, clean_text(txt)))
    doc.close()
    return pages
