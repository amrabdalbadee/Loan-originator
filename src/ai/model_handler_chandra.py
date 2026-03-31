"""
model_handler.py
================
Handles loading and inference of the Chandra VLM (Vision Language Model)
for Arabic financial document extraction.

Chandra is built on Qwen3-VL and is loaded via the `chandra-ocr` package.
Supports two modes:
  - OCR mode: faithful HTML reproduction of tables/content
  - Layout mode: full-page OCR with labeled bounding-box blocks
"""

import time

import logging
from typing import List, Dict, Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
MODEL_CHECKPOINT = "datalab-to/chandra"
BBOX_SCALE = 1024  # Chandra normalizes bboxes to 0-1024

# Prompts are loaded lazily from the chandra package at runtime.
_NATIVE_OCR_PROMPT = None
_NATIVE_LAYOUT_PROMPT = None


def _get_native_prompt() -> str:
    """Return Chandra's native OCR prompt (the one it was trained on)."""
    global _NATIVE_OCR_PROMPT
    if _NATIVE_OCR_PROMPT is None:
        try:
            from chandra.prompts import OCR_PROMPT
            _NATIVE_OCR_PROMPT = OCR_PROMPT
        except ImportError:
            _NATIVE_OCR_PROMPT = "OCR this image to HTML."
    return _NATIVE_OCR_PROMPT


def _get_layout_prompt() -> str:
    """Return Chandra's native layout-OCR prompt with bounding boxes."""
    global _NATIVE_LAYOUT_PROMPT
    if _NATIVE_LAYOUT_PROMPT is None:
        try:
            from chandra.prompts import OCR_LAYOUT_PROMPT
            _NATIVE_LAYOUT_PROMPT = OCR_LAYOUT_PROMPT
        except ImportError:
            _NATIVE_LAYOUT_PROMPT = "OCR this image to HTML arranged as layout blocks."
    return _NATIVE_LAYOUT_PROMPT


# Sentinel values for prompt selection
ARABIC_TABLE_PROMPT = "USE_NATIVE"
LAYOUT_PROMPT = "USE_LAYOUT"


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available - falling back to CPU. Inference will be slow.")
    return torch.device("cpu")


def load_model(model_checkpoint: str = MODEL_CHECKPOINT):
    """
    Load the Chandra VLM (Qwen3-VL based) model and processor.

    Returns
    -------
    tuple[model, processor, device, float, float]
    """
    start_time = time.perf_counter()
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    device = get_device()

    logger.info("Loading Qwen3VLProcessor from %s ...", model_checkpoint)
    processor = Qwen3VLProcessor.from_pretrained(model_checkpoint)

    logger.info("Loading Qwen3VLForConditionalGeneration from %s ...", model_checkpoint)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=dtype,
        device_map=str(device) if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model = model.eval()
    model.processor = processor

    load_time = time.perf_counter() - start_time
    model_size_bytes = model.get_memory_footprint()
    model_size_gb = model_size_bytes / (1024**3)

    logger.info(
        "Model loaded (dtype=%s, device=%s, time=%.2fs, size=%.2fGB)",
        dtype,
        device,
        load_time,
        model_size_gb,
    )
    return model, processor, device, load_time, model_size_gb



def _scale_to_fit(
    img: Image.Image,
    max_size=(3072, 2048),
    min_size=(28, 28),
) -> Image.Image:
    """Resize image to fit within Chandra's expected input range."""
    import math

    width, height = img.size
    if width == 0 or height == 0:
        return img

    max_w, max_h = max_size
    min_w, min_h = min_size
    current_pixels = width * height
    max_pixels = max_w * max_h
    min_pixels = min_w * min_h

    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        new_w = math.floor(width * scale)
        new_h = math.floor(height * scale)
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5
        new_w = math.ceil(width * scale)
        new_h = math.ceil(height * scale)
    else:
        return img

    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def _run_inference(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 8192,
) -> str:
    """
    Core inference: send image + prompt to the Chandra VLM and return
    the raw generated text.
    """
    from qwen_vl_utils import process_vision_info

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = _scale_to_fit(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        padding_side="left",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return result.strip()


def extract_table_from_image(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    prompt: str = "USE_NATIVE",
    max_new_tokens: int = 8192,
) -> str:
    """
    Run Chandra OCR on a single image and return raw HTML.
    If prompt is "USE_NATIVE", uses Chandra's trained OCR prompt.
    If prompt is "USE_LAYOUT", uses Chandra's layout prompt.
    """
    try:
        if prompt == "USE_NATIVE":
            prompt = _get_native_prompt()
        elif prompt == "USE_LAYOUT":
            prompt = _get_layout_prompt()

        return _run_inference(image, model, processor, device, prompt, max_new_tokens)

    except Exception as exc:
        logger.exception("VLM inference failed: %s", exc)
        return f"[ERROR] VLM inference failed: {exc}"


def extract_page_layout(
    image: Image.Image,
    model,
    processor,
    device: torch.device,
    max_new_tokens: int = 12384,
) -> str:
    """
    Run Chandra's layout-OCR on a full page image.
    Returns raw HTML with <div data-bbox="..." data-label="..."> blocks.
    """
    try:
        prompt = _get_layout_prompt()
        return _run_inference(image, model, processor, device, prompt, max_new_tokens)
    except Exception as exc:
        logger.exception("Layout extraction failed: %s", exc)
        return f"[ERROR] Layout extraction failed: {exc}"


def group_table_with_context(
    blocks: List[Dict[str, Any]],
    page_image: Image.Image,
    context_labels: tuple = ("Section-Header", "Text", "Caption", "Footnote", "Page-Header"),
) -> List[Dict[str, Any]]:
    """
    For every Table block, find context blocks (Section-Header, Text, etc.)
    that are directly to the LEFT or RIGHT of the table with vertical overlap.

    No above/below merging — only side blocks are captured.

    Each right-side block is classified as either:
      - WIDE  (width > table_width): a section title → used as column name
      - NARROW (width <= table_width): a per-row label → used as row data

    Each Table block is augmented with:
      - ``context_right``  : all blocks to the right with vertical overlap
      - ``context_left``   : all blocks to the left with vertical overlap
      - ``context_blocks`` : union of right + left
      - ``merged_bbox``    : bbox spanning table + side context
      - ``merged_image``   : PIL crop of merged_bbox
      - ``merged_html``    : concatenated HTML
      - ``merged_text``    : concatenated plain text
    """
    width, height = page_image.size
    grouped: List[Dict[str, Any]] = []

    for i, block in enumerate(blocks):
        if block["label"] != "Table":
            grouped.append(block)
            continue

        t_x0, t_y0, t_x1, t_y1 = block["bbox"]

        context_right: List[Dict[str, Any]] = []
        context_left:  List[Dict[str, Any]] = []

        logger.info(
            "TABLE #%d bbox=[%d,%d,%d,%d] — scanning %d candidate blocks",
            i, t_x0, t_y0, t_x1, t_y1, len(blocks) - 1,
        )

        for j, other in enumerate(blocks):
            if i == j or other["label"] not in context_labels:
                continue

            o_x0, o_y0, o_x1, o_y1 = other["bbox"]

            # Vertical overlap: the other block shares the table's Y range
            v_overlap = not (o_y1 < t_y0 or o_y0 > t_y1)

            # Is the other block entirely to the RIGHT of the table?
            is_right = o_x0 >= t_x1 and v_overlap
            # Is the other block entirely to the LEFT of the table?
            is_left  = o_x1 <= t_x0 and v_overlap

            logger.info(
                "  candidate [%s] bbox=[%d,%d,%d,%d] "
                "v_overlap=%s is_right=%s is_left=%s",
                other["label"], o_x0, o_y0, o_x1, o_y1,
                v_overlap, is_right, is_left,
            )

            if is_right:
                logger.info("    → MATCHED as RIGHT")
                context_right.append(other)
            elif is_left:
                logger.info("    → MATCHED as LEFT")
                context_left.append(other)
            else:
                logger.info("    → NOT matched")

        all_context = context_right + context_left

        # Build merged bbox (only left/right context — no above/below)
        mx0, my0, mx1, my1 = t_x0, t_y0, t_x1, t_y1
        for ctx in all_context:
            cx0, cy0, cx1, cy1 = ctx["bbox"]
            mx0 = min(mx0, cx0)
            my0 = min(my0, cy0)
            mx1 = max(mx1, cx1)
            my1 = max(my1, cy1)

        pad = 10
        merged_bbox = [
            max(0,      mx0 - pad),
            max(0,      my0 - pad),
            min(width,  mx1 + pad),
            min(height, my1 + pad),
        ]
        merged_image = page_image.crop(merged_bbox)

        def _html(b: Dict[str, Any]) -> str:
            return b.get("content_html", "") or ""

        def _text(b: Dict[str, Any]) -> str:
            return b.get("content_text", "") or ""

        merged_html = "\n".join(filter(None, [
            "\n".join(_html(b) for b in context_right),
            _html(block),
            "\n".join(_html(b) for b in context_left),
        ])).strip()

        merged_text = "\n".join(filter(None, [
            "\n".join(_text(b) for b in context_right),
            _text(block),
            "\n".join(_text(b) for b in context_left),
        ])).strip()

        grouped.append({
            **block,
            "context_blocks": all_context,
            "context_right":  context_right,
            "context_left":   context_left,
            "merged_bbox":    merged_bbox,
            "merged_image":   merged_image,
            "merged_html":    merged_html,
            "merged_text":    merged_text,
        })

    return grouped


def _expand_block_to_sublines(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a block whose ``content_html`` contains multiple text lines into
    one virtual sub-block per line with evenly interpolated Y positions.

    Three strategies are tried in order:
      1. ``<p>`` / ``<li>`` block-level tags
      2. ``<br>`` tags used as line separators
      3. Plain newline-separated text nodes (no HTML tags between lines)

    Falls back to returning the original block unchanged when only a single
    meaningful text line is found.
    """
    try:
        from bs4 import BeautifulSoup, NavigableString, Tag

        html  = block.get("content_html", "")
        soup  = BeautifulSoup(html, "html.parser")
        SENTINEL = "\x00"

        # ── Strategy 1: explicit block-level tags (<p>, <li>) ────────────
        block_tags = soup.find_all(["p", "li"])
        if len(block_tags) > 1:
            lines = [t.get_text(strip=True) for t in block_tags if t.get_text(strip=True)]
            if len(lines) > 1:
                return _make_subblocks(block, lines)

        # ── Strategy 2: <br> as line separator ───────────────────────────
        for br in soup.find_all("br"):
            br.replace_with(SENTINEL)

        # Collect top-level text, skipping <table> subtrees
        parts: List[str] = []
        for child in soup.children:
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag) and child.name not in (
                "table", "thead", "tbody", "tr", "td", "th"
            ):
                parts.append(child.get_text(separator=SENTINEL))

        br_text = SENTINEL.join(parts)
        lines = [seg.strip() for seg in br_text.split(SENTINEL) if seg.strip()]
        if len(lines) > 1:
            return _make_subblocks(block, lines)

        # ── Strategy 3: plain newlines between text nodes ─────────────────
        # Re-parse without the sentinel replacements
        soup2 = BeautifulSoup(html, "html.parser")
        nl_parts: List[str] = []
        for child in soup2.children:
            if isinstance(child, NavigableString):
                nl_parts.append(str(child))
            elif isinstance(child, Tag) and child.name not in (
                "table", "thead", "tbody", "tr", "td", "th"
            ):
                nl_parts.append(child.get_text(separator="\n"))

        nl_text = "\n".join(nl_parts)
        lines = [seg.strip() for seg in nl_text.splitlines() if seg.strip()]
        if len(lines) > 1:
            return _make_subblocks(block, lines)

    except Exception as exc:
        logger.debug("_expand_block_to_sublines: failed (%s), returning original", exc)

    return [block]


def _make_subblocks(block: Dict[str, Any], lines: List[str]) -> List[Dict[str, Any]]:
    """
    Given a list of text lines and the parent block's bbox, return one
    virtual sub-block dict per line with evenly interpolated Y positions.
    """
    b_x0, b_y0, b_x1, b_y1 = block["bbox"]
    block_height = max(b_y1 - b_y0, 1)
    line_height  = block_height / len(lines)

    sub_blocks = []
    for i, text in enumerate(lines):
        sub_y0 = b_y0 + i * line_height
        sub_y1 = b_y0 + (i + 1) * line_height
        sub_blocks.append({
            "content_text": text,
            "bbox": [b_x0, sub_y0, b_x1, sub_y1],
        })

    logger.info(
        "_expand_block_to_sublines: split 1 block into %d sub-lines %s",
        len(lines), lines,
    )
    return sub_blocks


def merge_row_labels_into_dataframe(
    df: "pd.DataFrame",
    block: Dict[str, Any],
) -> "pd.DataFrame":
    """
    Merge side-context row labels into ``df`` as a new rightmost column.

    Classification of right-side blocks by width vs table width:
      - WIDE  (block_width > table_width): section title  → column header name
      - NARROW (block_width <= table_width): per-row label → data in column

    Each narrow block is first expanded via ``_expand_block_to_sublines()``
    so that a single Chandra block containing multiple ``<p>``-separated
    lines is treated as individual per-row labels rather than one merged cell.

    Each (sub-)block is then Y-aligned to the closest table row using the
    row's estimated Y-centre (even-split over table height).

    The column is appended as the LAST column (rightmost — RTL reading order).
    """
    import pandas as pd

    t_x0, t_y0, t_x1, t_y1 = block.get("bbox", [0, 0, 0, 0])
    table_width  = max(t_x1 - t_x0, 1)
    table_height = max(t_y1 - t_y0, 1)
    n_rows = len(df)

    if n_rows == 0:
        return df

    # Use right-side blocks (RTL) or left-side blocks (LTR)
    side_blocks = block.get("context_right", []) or block.get("context_left", [])
    if not side_blocks:
        return df

    # ── Separate wide (title) vs narrow (row label) blocks ───────────────
    wide_blocks   = [b for b in side_blocks if (b["bbox"][2] - b["bbox"][0]) > table_width]
    narrow_blocks = [b for b in side_blocks if (b["bbox"][2] - b["bbox"][0]) <= table_width]

    # Column name: text of the topmost wide block (the section title)
    wide_sorted = sorted(wide_blocks, key=lambda b: b["bbox"][1])
    col_name = wide_sorted[0].get("content_text", "البيان").strip() if wide_sorted else "البيان"
    if len(col_name) > 60:
        col_name = col_name[:57] + "..."

    if not narrow_blocks:
        logger.info("merge_row_labels: no narrow row-label blocks found")
        return df

    # ── Expand any multi-line blocks into individual sub-lines ───────────
    expanded: List[Dict[str, Any]] = []
    for nb in narrow_blocks:
        expanded.extend(_expand_block_to_sublines(nb))

    # ── Y-align each (sub-)block to the closest table row ────────────────
    row_height  = table_height / n_rows
    row_centers = [t_y0 + (r + 0.5) * row_height for r in range(n_rows)]

    labels = [""] * n_rows
    for nb in expanded:
        nb_cy = (nb["bbox"][1] + nb["bbox"][3]) / 2
        closest_row = min(range(n_rows), key=lambda r: abs(row_centers[r] - nb_cy))
        text = nb.get("content_text", "").strip()
        if labels[closest_row]:
            labels[closest_row] += " " + text
        else:
            labels[closest_row] = text

    logger.info(
        "merge_row_labels: table bbox=%s col='%s' narrow=%d expanded=%d rows=%d labels=%s",
        block.get("bbox"), col_name, len(narrow_blocks), len(expanded), n_rows, labels,
    )

    df = df.copy()
    df[col_name] = labels   # append as last (rightmost) column
    return df


def parse_layout_blocks(raw_html: str, page_image: Image.Image) -> List[Dict[str, Any]]:
    """
    Parse Chandra's layout HTML output into a list of structured blocks.

    Each block has:
      - label: str (Table, Text, Section-Header, etc.)
      - bbox: [x0, y0, x1, y1] in pixel coordinates
      - content_html: str (inner HTML of the block)
      - content_text: str (plain text)
    """
    from bs4 import BeautifulSoup
    import json

    soup = BeautifulSoup(raw_html, "html.parser")

    # Find all divs with data-label (layout blocks), regardless of nesting
    top_divs = soup.find_all("div", attrs={"data-label": True})
    if not top_divs:
        # Fallback: try top-level divs
        top_divs = soup.find_all("div", recursive=False)

    width, height = page_image.size
    w_scale = width / BBOX_SCALE
    h_scale = height / BBOX_SCALE

    blocks = []
    for div in top_divs:
        label = div.get("data-label", "Unknown")

        # Parse bbox
        bbox_raw = div.get("data-bbox", "0 0 1 1")
        try:
            bbox = json.loads(bbox_raw)
        except (json.JSONDecodeError, TypeError):
            try:
                bbox = list(map(int, bbox_raw.split()))
            except Exception:
                bbox = [0, 0, 1, 1]

        # Scale to pixel coordinates
        bbox_px = [
            max(0, int(bbox[0] * w_scale)),
            max(0, int(bbox[1] * h_scale)),
            min(int(bbox[2] * w_scale), width),
            min(int(bbox[3] * h_scale), height),
        ]

        content_html = str(div.decode_contents())
        content_text = div.get_text(separator=" ", strip=True)

        blocks.append({
            "label": label,
            "bbox": bbox_px,
            "content_html": content_html,
            "content_text": content_text,
        })

    return blocks