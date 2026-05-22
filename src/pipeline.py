"""End-to-end: input PDF → cleaned PDF.

Usage:
    python -m src.pipeline input/AlisaDocs.pdf output/AlisaDocs_clean.pdf
    python -m src.pipeline input/AlisaDocs.pdf output/AlisaDocs_clean.pdf --debug
    python -m src.pipeline input/AlisaDocs.pdf output/AlisaDocs_clean.pdf --detector paddle
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

from src.detect import detect_text_boxes
from src.erase import erase_text
from src.pdf_io import load_input, save_pdf


def process_page(image, page_idx: int, detector: str, debug_dir: Path | None):
    print(f"[page {page_idx}] detecting text ({detector})…", flush=True)
    boxes = detect_text_boxes(image, prefer=detector)
    print(f"[page {page_idx}] {len(boxes)} text region(s)", flush=True)

    if not boxes:
        print(f"[page {page_idx}] no text — returning original", flush=True)
        return image

    print(f"[page {page_idx}] erasing…", flush=True)
    cleaned, mask = erase_text(image, boxes)

    if debug_dir is not None:
        cv2.imwrite(str(debug_dir / f"page_{page_idx}_mask.png"), mask)
        cv2.imwrite(str(debug_dir / f"page_{page_idx}_clean.png"), cleaned)

    return cleaned


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Erase text from scanned documents.")
    parser.add_argument("input", type=Path, help="Input PDF or image")
    parser.add_argument("output", type=Path, help="Output PDF path")
    parser.add_argument("--detector", choices=["gemini", "paddle"], default="gemini",
                        help="Primary detector. Gemini falls back to Paddle automatically on error.")
    parser.add_argument("--debug", action="store_true", help="Save per-page mask + cleaned PNG to output/debug/")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    debug_dir: Path | None = None
    if args.debug:
        debug_dir = args.output.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input}…")
    pages = load_input(args.input)
    print(f"{len(pages)} page(s)")

    cleaned_pages = []
    for i, page in enumerate(pages, start=1):
        cleaned_pages.append(process_page(page, i, args.detector, debug_dir))
        gc.collect()

    print(f"Saving {args.output}…")
    save_pdf(cleaned_pages, args.output)
    print(f"Done: {args.output}")
    if debug_dir:
        print(f"Debug images: {debug_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
