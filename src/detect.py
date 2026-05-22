"""Text-region detection.

Primary: Gemini 2.0 Flash — classifies every region as printed_text,
handwritten_text, signature, or stamp. Only the text categories are returned
as "erase me" bounding boxes; signatures and stamps are preserved by omission.

Fallback: PaddleOCR — pure text detector. Signatures and stamps are skipped
automatically because OCR doesn't recognise them as recognisable text (or it
returns them with low confidence, which we filter out).

Return shape (both paths):
    list[tuple[int, int, int, int]]   # (x1, y1, x2, y2) pixel boxes to erase
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


# ---------- Gemini path ----------

GEMINI_PROMPT = """Analyse this scanned document. Return ONLY a JSON array of objects, each with EXACTLY two keys:
  "type": one of "printed_text" | "handwritten_text" | "signature" | "stamp"
  "box_2d": [ymin, xmin, ymax, xmax] in 0–1000 normalised space

Group text into LARGE logical blocks (a full paragraph = ONE object, not one per line). Aim for at most ~60 objects total. Skip decorative watermarks/borders. Make boxes generous (5% padding). NO other keys, NO markdown, NO commentary — just the JSON array."""

GEMINI_MODEL_CASCADE = [
    "gemini-2.0-flash",        # highest free-tier quota
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

_EXHAUSTED: set[str] = set()


def _gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _gemini_image_part(image_bgr: np.ndarray):
    from google.genai import types
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png")


def _denormalize(boxes_0_1000: list[list[int]], width: int, height: int) -> list[BBox]:
    out: list[BBox] = []
    for box in boxes_0_1000:
        if not box or len(box) != 4:
            continue
        ymin, xmin, ymax, xmax = box
        x1 = int(xmin / 1000 * width)
        y1 = int(ymin / 1000 * height)
        x2 = int(xmax / 1000 * width)
        y2 = int(ymax / 1000 * height)
        x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append((x1, y1, x2, y2))
    return out


def _try_parse_json_array(raw: str) -> list:
    """Robust JSON-array parsing.

    Gemini occasionally returns truncated or near-malformed JSON when the
    response is very long (we've seen 100+ KB replies cut off mid-object).
    Strategy:
      1. Strip ``` fences.
      2. Try a clean parse.
      3. If that fails, truncate to the last complete `}` before the error
         and try again with a closing `]`.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1] if "```" in raw else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        last_close = raw[:e.pos].rfind("}")
        if last_close == -1:
            return []
        candidate = raw[: last_close + 1].rstrip(", \n\t") + "]"
        try:
            data = json.loads(candidate)
            print(f"  Gemini JSON truncated; recovered {len(data) if isinstance(data, list) else 0} elements", flush=True)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def detect_with_gemini(image_bgr: np.ndarray) -> list[BBox]:
    """Detect text regions via Gemini, return only printed/handwritten text boxes."""
    from google.genai import errors as genai_errors
    from google.genai import types

    client = _gemini_client()
    image_part = _gemini_image_part(image_bgr)

    def _call(model_name: str):
        return client.models.generate_content(
            model=model_name,
            contents=[GEMINI_PROMPT, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )

    response = None
    for model_name in GEMINI_MODEL_CASCADE:
        if model_name in _EXHAUSTED:
            continue
        try:
            response = _call(model_name)
            print(f"  → Gemini used: {model_name}", flush=True)
            break
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            msg = str(e)
            short = msg[:140].replace("\n", " ")
            if "404" in msg or "NOT_FOUND" in msg:
                print(f"  {model_name}: not available, skipping", flush=True)
                _EXHAUSTED.add(model_name)
                continue
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                m = re.search(r"retry in (\d+)", msg)
                wait = int(m.group(1)) + 1 if m else 0
                if wait and wait <= 10:
                    print(f"  {model_name}: rate-limited, waiting {wait}s…", flush=True)
                    time.sleep(wait)
                    try:
                        response = _call(model_name)
                        break
                    except Exception as e2:
                        print(f"  {model_name}: still failing ({str(e2)[:80]})", flush=True)
                print(f"  {model_name}: quota exhausted", flush=True)
                _EXHAUSTED.add(model_name)
                continue
            if "503" in msg or "UNAVAILABLE" in msg:
                print(f"  {model_name}: overloaded, waiting 8s…", flush=True)
                time.sleep(8)
                try:
                    response = _call(model_name)
                    break
                except Exception as e2:
                    print(f"  {model_name}: still overloaded ({str(e2)[:80]})", flush=True)
                    continue
            print(f"  {model_name}: error → {short}", flush=True)
            continue
        except Exception as e:
            # Catch genai SDK-level errors too (e.g. config validation)
            print(f"  {model_name}: unexpected → {str(e)[:140]}", flush=True)
            continue

    if response is None:
        raise RuntimeError("All Gemini models exhausted")

    data = _try_parse_json_array(response.text)
    if not data:
        raise RuntimeError("Gemini returned unparseable JSON")

    # Filter: keep ONLY printed_text and handwritten_text — those are erased.
    # Signatures and stamps are dropped here, so they stay untouched on the page.
    text_boxes_normalised: list[list[int]] = []
    for el in data:
        if not isinstance(el, dict):
            continue
        if el.get("type") in ("printed_text", "handwritten_text"):
            box = el.get("box_2d") or el.get("bbox")
            if box and len(box) == 4:
                text_boxes_normalised.append(box)

    h, w = image_bgr.shape[:2]
    return _denormalize(text_boxes_normalised, w, h)


# ---------- PaddleOCR path ----------

_PADDLE = None


def _paddle():
    """Lazy singleton — PaddleOCR is slow to import. Compatible with both
    PaddleOCR 2.x (use_angle_cls) and 3.x (different ctor)."""
    global _PADDLE
    if _PADDLE is None:
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        from paddleocr import PaddleOCR
        try:
            # 2.x API
            _PADDLE = PaddleOCR(use_angle_cls=True, lang="ru")
        except (TypeError, ValueError):
            # 3.x API — different params
            _PADDLE = PaddleOCR(lang="ru")
    return _PADDLE


def _paddle_ocr_call(ocr, image_rgb):
    """Wrap the OCR call to handle both 2.x (.ocr) and 3.x (.predict) APIs."""
    # 2.x returns: [[ [poly], (text, conf) ], …]
    # 3.x returns: list of dict with 'rec_polys', 'rec_texts', 'rec_scores'
    if hasattr(ocr, "predict"):
        try:
            res = ocr.predict(image_rgb)
        except Exception:
            res = None
        if res:
            out = []
            for page in res:
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                scores = page.get("rec_scores") or [1.0] * len(polys)
                for poly, score in zip(polys, scores):
                    out.append((poly, score))
            return ("v3", out)
    # 2.x
    res = ocr.ocr(image_rgb, cls=True)
    if not res or not res[0]:
        return ("v2", [])
    return ("v2", res[0])


def detect_with_paddle(image_bgr: np.ndarray, min_conf: float = 0.5) -> list[BBox]:
    """Fallback detector. Signatures usually fail OCR or come back with very low
    confidence — we filter those out, leaving them on the page."""
    ocr = _paddle()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    api_ver, lines = _paddle_ocr_call(ocr, image_rgb)
    boxes: list[BBox] = []
    for entry in lines:
        if api_ver == "v3":
            poly, conf = entry
        else:
            poly, (_text, conf) = entry[0], entry[1]
        if conf < min_conf:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append((x1, y1, x2, y2))
    return boxes


# ---------- Public entry point ----------

def detect_text_boxes(image_bgr: np.ndarray, prefer: str = "gemini") -> list[BBox]:
    """Try Gemini first, fall back to PaddleOCR if it fails."""
    if prefer == "gemini":
        try:
            return detect_with_gemini(image_bgr)
        except Exception as e:
            print(f"  Gemini failed ({e}). Falling back to PaddleOCR…", flush=True)
            return detect_with_paddle(image_bgr)
    return detect_with_paddle(image_bgr)
