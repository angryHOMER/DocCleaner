"""PDF ↔ image conversion utilities."""
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


def load_pdf_pages(pdf_path: Path, dpi: int = 200) -> list[np.ndarray]:
    """Render every page of a PDF to a BGR numpy array."""
    pages: list[np.ndarray] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            pages.append(arr)
    finally:
        doc.close()
    return pages


def load_input(path: Path) -> list[np.ndarray]:
    """Load PDF or a single image file as a list of BGR pages."""
    if path.suffix.lower() == ".pdf":
        return load_pdf_pages(path)
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return [img]


def save_pdf(images: list[np.ndarray], out_path: Path) -> None:
    """Pack a list of BGR images into a single PDF, one page each."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    try:
        for img in images:
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                raise RuntimeError("PNG encode failed")
            h, w = img.shape[:2]
            page = doc.new_page(width=w, height=h)
            page.insert_image(fitz.Rect(0, 0, w, h), stream=buf.tobytes())
        doc.save(out_path)
    finally:
        doc.close()
