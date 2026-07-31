"""
Vision tool implementation for TTSP: image_zoom_in_tool.
"""
import os
import re
import math
import json
import tempfile
import traceback
from typing import Optional, Tuple, List
from PIL import Image


# ============= IMAGE PROCESSING =============

def smart_resize(height: int, width: int, factor: int = 32,
                 min_pixels: int = 256 * 32 * 32,
                 max_pixels: int = 12845056) -> Tuple[int, int]:
    """Calculate new dimensions preserving aspect ratio within pixel budget."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def crop_and_resize_image(
    pil_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    dataset_name: Optional[str] = None,
    sample_name: Optional[str] = None,
    turn_count: int = 0,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    trace_idx: Optional[int] = None,
    save_image: bool = False,
) -> Tuple[Image.Image, str]:
    """Crop image by bbox [0-1000 normalized] and resize to appropriate dimensions."""
    img_width, img_height = pil_image.size
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox

    left  = max(0, rel_x1 / 1000.0 * img_width)
    top   = max(0, rel_y1 / 1000.0 * img_height)
    right = min(img_width,  rel_x2 / 1000.0 * img_width)
    bottom= min(img_height, rel_y2 / 1000.0 * img_height)

    # Validate & fix degenerate boxes
    if left >= right:
        left, right = min(left, right), max(left, right)
        if left == right:
            cx = left; left = max(0, cx - 16); right = min(img_width, cx + 16)
    if top >= bottom:
        top, bottom = min(top, bottom), max(top, bottom)
        if top == bottom:
            cy = top; top = max(0, cy - 16); bottom = min(img_height, cy + 16)

    # Expand if too small
    h, w = bottom - top, right - left
    if h < 32 or w < 32:
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        ar = max(h, w) / max(min(h, w), 1e-6)
        if ar > 10:
            if h > w:
                th, tw = max(32, h), max(32, h / 10)
            else:
                th, tw = max(32, w / 10), max(32, w)
            hh, hw = math.ceil(th * 0.5), math.ceil(tw * 0.5)
        else:
            ratio = 32 / min(h, w)
            hh = math.ceil(h * ratio * 0.5)
            hw = math.ceil(w * ratio * 0.5)
        nl, nt = max(0, math.floor(cx - hw)), max(0, math.floor(cy - hh))
        nr, nb = min(img_width, math.ceil(cx + hw)), min(img_height, math.ceil(cy + hh))
        nh, nw = nb - nt, nr - nl
        if nh >= 32 and nw >= 32 and max(nh, nw) / max(min(nh, nw), 1e-6) < 150:
            left, top, right, bottom = nl, nt, nr, nb

    cropped = pil_image.crop((left, top, right, bottom))
    new_w, new_h = smart_resize(int(right - left), int(bottom - top), factor=32)
    cropped = cropped.resize((new_w, new_h), resample=Image.BICUBIC)

    if save_image and dataset_name and sample_name:
        parts = [os.getcwd(), "zoom_in_img"]
        if model_name:
            parts.extend([model_name, dataset_name])
        else:
            parts.append(dataset_name)
        parts.append(sample_name)
        folder = os.path.join(*parts)
        os.makedirs(folder, exist_ok=True)
        fname = f"trace{trace_idx}_turn{turn_count}.png" if trace_idx is not None else f"{turn_count}.png"
        path = os.path.join(folder, fname)
    else:
        suffix = f"_trace{trace_idx}_turn{turn_count}.png" if trace_idx is not None else f"_turn{turn_count}.png"
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name

    cropped.save(path)
    return cropped, path


def cleanup_zoom_image(image_path: str):
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception:
            pass


# ============= TOOL PARSING =============

def parse_tool_calls_from_response(response_text: str) -> List[Tuple[str, str]]:
    """Parse all <tool_call> JSON blocks from model response."""
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_calls = []
    for match in re.finditer(pattern, response_text, re.DOTALL):
        try:
            data = json.loads(match.group(1))
            name = data.get('name')
            args = data.get('arguments', {})
            if name:
                tool_calls.append((name, json.dumps(args, ensure_ascii=False)))
        except json.JSONDecodeError:
            continue
    return tool_calls


# ============= TOOL EXECUTION =============

def execute_tool_call(
    tool_name: str,
    tool_args_json: str,
    all_image_paths: list,
    model_short_name: str = "",
    dataset_name: Optional[str] = None,
    sample_name: Optional[str] = None,
    turn_count: int = 0,
    run_id: Optional[str] = None,
    trace_idx: Optional[int] = None,
    save_image: bool = False,
) -> Tuple[bool, Optional[str], Optional[Image.Image], str, Optional[list]]:
    """Execute image_zoom_in_tool. Returns (success, path, pil, desc, bbox)."""
    if tool_name != "image_zoom_in_tool" or not all_image_paths:
        return False, None, None, "", None
    try:
        args = json.loads(tool_args_json)
        bbox = args.get("bbox_2d")
        label = args.get("label", "region")
        img_idx = args.get("img_idx", 0)

        if not bbox or len(bbox) != 4:
            return False, None, None, "", None
        if img_idx < 0 or img_idx >= len(all_image_paths):
            return False, None, None, "", None

        img_path = all_image_paths[img_idx]
        if img_path.startswith('file://'):
            img_path = img_path[len('file://'):]
        if not os.path.exists(img_path):
            return False, None, None, "", None

        src = Image.open(img_path)
        cropped, cropped_path = crop_and_resize_image(
            src, tuple(bbox), dataset_name, sample_name,
            turn_count, model_short_name, run_id, trace_idx, save_image
        )
        desc = f"Zoomed in on: {label} (image {img_idx})"
        return True, cropped_path, cropped, desc, bbox
    except Exception:
        traceback.print_exc()
        return False, None, None, "", None
