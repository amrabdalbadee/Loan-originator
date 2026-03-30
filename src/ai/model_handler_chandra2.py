"""
model_handler_chandra2.py
=========================
Handles loading and inference of the **Chandra OCR 2** VLM.

Chandra OCR 2 is built on Qwen 3.5 and is loaded via the ``chandra-ocr``
package (>=0.2.0).  It uses ``AutoModelForImageTextToText`` +
``chandra.model.hf.generate_hf`` for inference.

Supports two prompt types via ``BatchInputItem.prompt_type``:
  - ``"ocr"``        : faithful OCR reproduction (tables / content)
  - ``"ocr_layout"`` : full-page OCR with labeled bounding-box blocks

Public API used by app.py:
  - load_model()
  - _get_native_prompt()
  - extract_table_from_image()
  - extract_page_layout()
  - parse_layout_blocks()
  - group_table_with_context()   (re-exported from model_handler)
  - merge_row_labels_into_dataframe()  (re-exported from model_handler)
"""

import logging
import time
from typing import Any, Dict, List

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "datalab-to/chandra-ocr-2"
BBOX_SCALE = 1024  # Chandra normalizes bboxes to 0-1024

# Sentinel values for prompt selection — same convention as model_handler.py
ARABIC_TABLE_PROMPT = "USE_NATIVE"
LAYOUT_PROMPT = "USE_LAYOUT"

# Lazy prompt strings
_NATIVE_OCR_PROMPT = None
_NATIVE_LAYOUT_PROMPT = None


# ---------------------------------------------------------------------------
# Re-export helpers from Chandra v1 that are still needed
# ---------------------------------------------------------------------------
try:
    from .model_handler import (  # noqa: E402, F401
        group_table_with_context,
        merge_row_labels_into_dataframe,
    )
except ImportError:
    # If model_handler is not available, we define placeholders or skip
    logger.warning("model_handler not found. Helper functions will not be available.")
    def group_table_with_context(*args, **kwargs):
        raise NotImplementedError("group_table_with_context requires model_handler.")
    def merge_row_labels_into_dataframe(*args, **kwargs):
        raise NotImplementedError("merge_row_labels_into_dataframe requires model_handler.")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _get_native_prompt() -> str:
    """Return Chandra OCR 2's native OCR prompt (trained prompt)."""
    global _NATIVE_OCR_PROMPT
    if _NATIVE_OCR_PROMPT is None:
        try:
            from chandra.prompts import PROMPT_MAPPING
            _NATIVE_OCR_PROMPT = PROMPT_MAPPING["ocr"]
        except (ImportError, KeyError):
            _NATIVE_OCR_PROMPT = "OCR this image to HTML."
    return _NATIVE_OCR_PROMPT


def _get_layout_prompt() -> str:
    """Return Chandra OCR 2's layout-OCR prompt."""
    global _NATIVE_LAYOUT_PROMPT
    if _NATIVE_LAYOUT_PROMPT is None:
        try:
            from chandra.prompts import PROMPT_MAPPING
            _NATIVE_LAYOUT_PROMPT = PROMPT_MAPPING["ocr_layout"]
        except (ImportError, KeyError):
            _NATIVE_LAYOUT_PROMPT = (
                "OCR this image to HTML arranged as layout blocks."
            )
    return _NATIVE_LAYOUT_PROMPT


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning(
        "CUDA not available — falling back to CPU. "
        "Inference will be slow."
    )
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_checkpoint: str = MODEL_CHECKPOINT):
    """
    Load the Chandra OCR 2 model and processor.

    Uses ``AutoModelForImageTextToText`` + ``AutoProcessor`` from
    HuggingFace Transformers.

    Returns
    -------
    tuple[model, processor, device, float, float]
        (model, processor, device, load_time_seconds, model_size_gb)
    """
    start_time = time.perf_counter()
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    device = get_device()
    dtype = (
        torch.bfloat16 if device.type == "cuda" else torch.float32
    )

    logger.info(
        "Loading Chandra OCR 2 model from %s …", model_checkpoint
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_checkpoint,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_checkpoint)
    processor.tokenizer.padding_side = "left"
    model.processor = processor

    load_time = time.perf_counter() - start_time
    model_size_bytes = model.get_memory_footprint()
    model_size_gb = model_size_bytes / (1024 ** 3)

    logger.info(
        "Chandra OCR 2 loaded "
        "(dtype=%s, device=%s, time=%.2fs, size=%.2fGB)",
        dtype, device, load_time, model_size_gb,
    )
    return model, processor, device, load_time, model_size_gb


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _run_inference_raw(
    image: Image.Image,
    model,
    prompt_type: str = "ocr",
    max_tokens: int = 8192,
) -> str:
    """
    Run Chandra OCR 2 inference and return the **raw** model output
    (HTML with ``<div data-bbox data-label>`` structure for layout,
    or plain HTML for OCR).

    Uses ``chandra.model.hf.generate_hf`` with ``BatchInputItem``.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem

    batch = [
        BatchInputItem(
            image=image,
            prompt_type=prompt_type,
        )
    ]

    # NOTE: the parameter is max_output_tokens, NOT max_new_tokens
    result = generate_hf(
        batch, model, max_output_tokens=max_tokens
    )[0]

    logger.info(
        "Chandra OCR 2 raw output (%s): %d chars, %d tokens",
        prompt_type, len(result.raw), result.token_count,
    )
    return result.raw.strip()


def _run_inference_ocr(
    image: Image.Image,
    model,
    max_tokens: int = 8192,
) -> str:
    """
    Run OCR inference and return **parsed markdown**.

    This is the appropriate mode for table extraction / single-element
    OCR where the caller wants clean text/markdown output.
    """
    raw = _run_inference_raw(image, model, "ocr", max_tokens)

    try:
        from chandra.output import parse_markdown
        return parse_markdown(raw).strip()
    except ImportError:
        logger.warning(
            "chandra.output.parse_markdown not available; "
            "returning raw output."
        )
        return raw


def _run_inference_layout(
    image: Image.Image,
    model,
    max_tokens: int = 12384,
) -> str:
    """
    Run layout-OCR inference and return the **raw HTML** containing
    ``<div data-bbox="..." data-label="...">`` blocks.

    The raw HTML is what ``parse_layout_blocks()`` expects — do NOT
    convert to markdown here.
    """
    return _run_inference_raw(
        image, model, "ocr_layout", max_tokens
    )


# ---------------------------------------------------------------------------
# Layout block parsing  (uses Chandra 2's own parser)
# ---------------------------------------------------------------------------

def parse_layout_blocks(
    raw_html: str,
    page_image: Image.Image,
) -> List[Dict[str, Any]]:
    """
    Parse Chandra OCR 2's layout HTML into structured blocks.

    Uses ``chandra.output.parse_layout`` which properly handles
    the ``<div data-bbox data-label>`` structure.

    Falls back to model_handler.parse_layout_blocks (v1) if
    the chandra package parser is unavailable.

    Each block has:
      - label: str
      - bbox: [x0, y0, x1, y1] in pixel coordinates
      - content_html: str
      - content_text: str
    """
    try:
        from chandra.output import parse_layout

        layout_blocks = parse_layout(raw_html, page_image)
        blocks = []
        for lb in layout_blocks:
            from bs4 import BeautifulSoup
            content_text = BeautifulSoup(
                lb.content, "html.parser"
            ).get_text(separator=" ", strip=True)

            blocks.append({
                "label": lb.label,
                "bbox": lb.bbox,
                "content_html": lb.content,
                "content_text": content_text,
            })

        logger.info(
            "parse_layout_blocks (chandra2 native): "
            "%d blocks found",
            len(blocks),
        )
        return blocks

    except ImportError:
        logger.warning(
            "chandra.output.parse_layout not available — "
            "falling back to v1 parser."
        )
        try:
            from .model_handler import parse_layout_blocks as _v1_parse
            return _v1_parse(raw_html, page_image)
        except ImportError:
            logger.error("v1 parser also unavailable.")
            return []


# ---------------------------------------------------------------------------
# Public API — app.py compatibility
# ---------------------------------------------------------------------------

def extract_table_from_image(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str = "USE_NATIVE",
    max_new_tokens: int = 8192,
) -> str:
    """
    Run Chandra OCR 2 on a single image and return parsed output.

    If prompt is ``"USE_NATIVE"``, uses ``"ocr"`` prompt type.
    If prompt is ``"USE_LAYOUT"``, uses ``"ocr_layout"`` type
    (returns raw HTML).
    """
    try:
        if prompt == "USE_LAYOUT":
            return _run_inference_layout(
                image, model, max_new_tokens
            )
        else:
            return _run_inference_ocr(
                image, model, max_new_tokens
            )
    except Exception as exc:
        logger.exception(
            "Chandra OCR 2 inference failed: %s", exc
        )
        return f"[ERROR] VLM inference failed: {exc}"


def extract_page_layout(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 12384,
) -> str:
    """
    Run Chandra OCR 2's layout-OCR on a full page image.

    Returns **raw HTML** with
    ``<div data-bbox="..." data-label="...">`` blocks.
    This output is fed into ``parse_layout_blocks()``.
    """
    try:
        return _run_inference_layout(
            image, model, max_new_tokens
        )
    except Exception as exc:
        logger.exception(
            "Chandra OCR 2 layout extraction failed: %s", exc
        )
        return f"[ERROR] Layout extraction failed: {exc}"
