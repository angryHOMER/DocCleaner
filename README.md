# DocCleaner

Stage 1 of a document-translation pipeline: erase printed and handwritten text from scanned/photographed pages while preserving signatures and stamps. Output is a clean PDF ready for the next stage (translation rendering).

## Pipeline

```
input.pdf → pages → detect_text → build_mask → inpaint → output.pdf
                        ↑
                Gemini 2.0 Flash (primary)
                PaddleOCR + PP-Structure (fallback)
```

- **Detection**: Gemini classifies every region as `printed_text`, `handwritten_text`, `signature`, or `stamp`. Only the text categories go into the erase mask. If Gemini fails or hits quota, PaddleOCR's text detector takes over (signatures/stamps are skipped automatically because OCR doesn't recognise them as text).
- **Inpainting**: LaMa (via `iopaint`) preserves textured backgrounds (watermarks, security patterns) far better than classical OpenCV inpainting.
- **Runtime**: optimised for Google Colab T4 GPU. Works on CPU too, just slower.

## Run on Colab (recommended)

Open `notebook.ipynb` in Colab, select **T4 GPU** runtime, run all cells.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then put your GEMINI_API_KEY in .env
python -m src.pipeline input/AlisaDocs.pdf output/AlisaDocs_clean.pdf
```

Get a free Gemini API key at https://aistudio.google.com/apikey.
