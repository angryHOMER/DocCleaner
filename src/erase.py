"""Erase text from a page using LaMa inpainting.

Build a pixel-level mask from text bounding boxes (so we don't paint over
clean background unnecessarily) and feed it to LaMa via iopaint.
"""
from __future__ import annotations

import gc
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BBox = tuple[int, int, int, int]

_MODEL = None
_DEVICE: str | None = None

_MAX_DIM_CPU = 1800   # 8 GB RAM-safe ceiling
_MAX_DIM_GPU = 4096


def _detect_device() -> str:
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _DEVICE = "mps"
        else:
            _DEVICE = "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def _lama_model():
    global _MODEL
    if _MODEL is None:
        from iopaint.model_manager import ModelManager
        device = _detect_device()
        print(f"  loading LaMa on {device.upper()} (first call may download ~200 MB)…", flush=True)
        _MODEL = ModelManager(name="lama", device=device)
        print(f"  LaMa ready on {device}", flush=True)
    return _MODEL


def build_mask(image_bgr: np.ndarray, text_boxes: list[BBox],
               margin: int = 14, bbox_expand_pct: float = 0.30) -> np.ndarray:
    """Pixel-level mask of text strokes inside the given bounding boxes.

    We don't mask the whole bbox — only the dark pixels inside it. That keeps
    the inpainter's job small (less area to fill) and preserves more of the
    original background texture.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Two thresholds combined: a strict global one + an adaptive one for faint text.
    _, strong = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=21, C=15,
    )
    text_pixels = cv2.bitwise_or(strong, adaptive)
    text_pixels = cv2.morphologyEx(text_pixels, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    allowed = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in text_boxes:
        bw, bh = x2 - x1, y2 - y1
        ex = max(margin, int(bw * bbox_expand_pct))
        ey = max(margin, int(bh * bbox_expand_pct))
        ax1 = max(0, x1 - ex)
        ay1 = max(0, y1 - ey)
        ax2 = min(w, x2 + ex)
        ay2 = min(h, y2 + ey)
        allowed[ay1:ay2, ax1:ax2] = 255

    mask = cv2.bitwise_and(text_pixels, allowed)
    # One small dilation to catch antialiased edges.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def lama_inpaint(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run LaMa on the masked region. Returns BGR image."""
    if cv2.countNonZero(mask) == 0:
        return image_bgr

    from iopaint.schema import HDStrategy, InpaintRequest

    model = _lama_model()
    device = _detect_device()
    max_dim = _MAX_DIM_GPU if device in ("cuda", "mps") else _MAX_DIM_CPU
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(1.0, max_dim / max(orig_w, orig_h))

    if scale < 1.0:
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        small_img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"    downscaled to {new_w}x{new_h} for memory safety", flush=True)
    else:
        small_img, small_mask = image_bgr, mask

    request = InpaintRequest(
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_crop_margin=128 if device in ("cuda", "mps") else 64,
        hd_strategy_resize_limit=max_dim,
        ldm_steps=20 if device in ("cuda", "mps") else 15,
        ldm_sampler="plms",
        zits_wireframe=True,
    )

    image_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    binary_mask = (small_mask > 0).astype(np.uint8) * 255
    result_rgb = model(image_rgb, binary_mask, request)
    result_bgr_small = cv2.cvtColor(result_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    del image_rgb, result_rgb
    gc.collect()

    if scale < 1.0:
        result_bgr = cv2.resize(result_bgr_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        result_bgr = result_bgr_small

    # Guarantee: every UNmasked pixel stays exactly original. Protects signatures,
    # stamps, watermarks from any colour drift LaMa might introduce around the
    # mask boundary.
    binary_mask_full = (mask > 0).astype(np.uint8) * 255
    mask_3ch = cv2.cvtColor(binary_mask_full, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    composed = (
        result_bgr.astype(np.float32) * mask_3ch
        + image_bgr.astype(np.float32) * (1.0 - mask_3ch)
    ).astype(np.uint8)
    return composed


def erase_text(image_bgr: np.ndarray, text_boxes: list[BBox]) -> tuple[np.ndarray, np.ndarray]:
    """Build mask + run LaMa. Returns (cleaned_image, mask)."""
    mask = build_mask(image_bgr, text_boxes)
    cleaned = lama_inpaint(image_bgr, mask)
    return cleaned, mask
