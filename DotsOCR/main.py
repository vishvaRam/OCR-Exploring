import json
import re

import pymupdf  # PyMuPDF
import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

# 1. Load Model & Processor
model_path = "./weights/DotsMOCR"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 2. Target Prompt (Exact copy)
prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def extract_json_from_response(text: str):
    """Extract and parse JSON whether the model returns a list [...] or an object {...}."""
    # Strip markdown fences if present
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate_str = code_match.group(1).strip() if code_match else text.strip()

    # Find the earliest opening bracket/brace and matching closing bracket/brace
    first_bracket = candidate_str.find("[")
    first_brace = candidate_str.find("{")

    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        last_bracket = candidate_str.rfind("]")
        if last_bracket != -1:
            candidate_str = candidate_str[first_bracket : last_bracket + 1]
    elif first_brace != -1:
        last_brace = candidate_str.rfind("}")
        if last_brace != -1:
            candidate_str = candidate_str[first_brace : last_brace + 1]

    return json.loads(candidate_str)


def draw_bboxes_on_image(pil_img: Image.Image, layout_data) -> Image.Image:
    """Draw bounding boxes and category labels on the PIL image."""
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)
    width, height = draw_img.size

    # Unpack elements regardless of top-level type
    if isinstance(layout_data, list):
        elements = layout_data
    elif isinstance(layout_data, dict):
        elements = (
            layout_data.get("elements")
            or layout_data.get("layout")
            or layout_data.get("items")
            or []
        )
    else:
        elements = []

    # Distinct colors per layout category
    color_map = {
        "Title": "red",
        "Section-header": "orange",
        "Text": "blue",
        "List-item": "cyan",
        "Table": "green",
        "Picture": "purple",
        "Formula": "magenta",
        "Caption": "gold",
        "Footnote": "gray",
        "Page-header": "brown",
        "Page-footer": "brown",
    }

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for item in elements:
        bbox = item.get("bbox")
        category = item.get("category", "Unknown")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Auto-scale normalized 0-1000 coordinates (DotsMOCR / Qwen standard)
        if max(x1, y1, x2, y2) <= 1000 and max(width, height) > 1000:
            x1, x2 = (x1 / 1000) * width, (x2 / 1000) * width
            y1, y2 = (y1 / 1000) * height, (y2 / 1000) * height
        elif max(x1, y1, x2, y2) <= 1.0:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height

        # Ensure correct box boundaries
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        color = color_map.get(category, "lime")

        # Draw box
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

        # Draw category tag
        label_height = 14
        label_width = len(category) * 7 + 4
        tag_top = max(0, top - label_height)
        draw.rectangle([left, tag_top, left + label_width, tag_top + label_height], fill=color)
        draw.text((left + 2, tag_top), category, fill="white", font=font)

    return draw_img


def process_pdf(pdf_path: str, output_prefix: str = "page"):
    doc = pymupdf.open(pdf_path)

    for page_idx in range(len(doc)):
        print(f"Processing page {page_idx + 1}/{len(doc)}...")
        page = doc[page_idx]

        # Render PDF page to PIL Image at 200 DPI
        zoom = 200 / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=24000)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        try:
            layout_data = extract_json_from_response(output_text)
            annotated_img = draw_bboxes_on_image(pil_image, layout_data)

            # Save annotated image
            out_img_path = f"{output_prefix}_{page_idx + 1}_annotated.png"
            annotated_img.save(out_img_path)
            print(f"Saved: {out_img_path}")

            # Save raw extracted layout JSON
            out_json_path = f"{output_prefix}_{page_idx + 1}_layout.json"
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to parse layout on page {page_idx + 1}: {e}")
            print(f"Raw output:\n{output_text}")


if __name__ == "__main__":
    process_pdf("Docs/pan1.pdf", output_prefix="output_page")