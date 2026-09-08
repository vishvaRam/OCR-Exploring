import json
import os

import pymupdf  # PyMuPDF
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
from PIL import Image, ImageDraw, ImageFont


def get_default_font():
    """Attempt to load a standard readable font, falling back to PIL default."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", 14)
        except Exception:
            return ImageFont.load_default()


def draw_predictions_on_image(pil_img: Image.Image, predictions: list) -> Image.Image:
    """Draw bounding boxes and recognized text on the PIL image."""
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)
    font = get_default_font()
    img_w, img_h = draw_img.size

    for item in predictions:
        # Resolve bounding coordinates based on available keys
        if item.get("quad"):
            # item["quad"] format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            quad = item["quad"]
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
        elif all(k in item for k in ("left", "right", "upper", "lower")):
            left = item["left"]
            right = item["right"]
            top = item["upper"]
            bottom = item["lower"]
        elif "bbox" in item:
            left, top, right, bottom = item["bbox"]
        else:
            continue

        # Scale coordinates if model returns normalized values (0.0 - 1.0)
        if max(left, right, top, bottom) <= 1.0:
            left *= img_w
            right *= img_w
            top *= img_h
            bottom *= img_h

        # Enforce canonical rectangle ordering
        x1, x2 = sorted([left, right])
        y1, y2 = sorted([top, bottom])

        # Draw bounding rectangle
        box_color = "#00FF66"
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

        # Draw detected text label if present
        text = item.get("text", "").strip()
        if text:
            label_text = text if len(text) <= 30 else f"{text[:27]}..."
            bbox_text = draw.textbbox((x1, y1), label_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]

            tag_top = max(0, y1 - text_h - 4)
            draw.rectangle(
                [x1, tag_top, x1 + text_w + 4, tag_top + text_h + 4],
                fill=box_color,
            )
            draw.text((x1 + 2, tag_top + 1), label_text, fill="black", font=font)

    return draw_img


def process_pdf_with_nemotron(
    pdf_path: str,
    output_dir: str = "output_nemotron",
    dpi: int = 200,
    merge_level: str = "word",
    skip_relational: bool = True,
):
    """Processes each page of a PDF with NemotronOCRV2 and saves annotated outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OCR Engine
    print("Loading NemotronOCRV2 engine...")
    ocr_engine = NemotronOCRV2(skip_relational=skip_relational)

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    print(f"Loaded '{pdf_path}' ({total_pages} pages).")

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        print(f"\n--- Processing Page {page_num}/{total_pages} ---")
        page = doc[page_idx]

        # Render PDF page to a temporary image path (required by file-path inputs)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        temp_img_path = os.path.join(output_dir, f"_temp_page_{page_num}.png")
        pix.save(temp_img_path)

        try:
            # Run inference
            predictions = ocr_engine(temp_img_path, merge_level=merge_level)

            # Convert result into serializable format if custom objects are returned
            serializable_preds = []
            for item in predictions:
                if hasattr(item, "__dict__"):
                    serializable_preds.append(item.__dict__)
                elif isinstance(item, dict):
                    serializable_preds.append(item)
                else:
                    serializable_preds.append(str(item))

            # Annotate image
            with Image.open(temp_img_path) as pil_img:
                annotated_img = draw_predictions_on_image(pil_img, serializable_preds)

                # Save annotated image
                out_img_path = os.path.join(
                    output_dir, f"page_{page_num}_annotated.png"
                )
                annotated_img.save(out_img_path)
                print(f"Saved annotated image: {out_img_path}")

            # Save prediction metadata JSON
            out_json_path = os.path.join(output_dir, f"page_{page_num}_ocr.json")
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_preds, f, indent=2, ensure_ascii=False)
            print(f"Saved OCR JSON: {out_json_path}")

        finally:
            # Clean up intermediate unannotated render
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    print("\nAll pages processed successfully.")


if __name__ == "__main__":
    pdf_file = "Docs/pan3.pdf"
    process_pdf_with_nemotron(
        pdf_path=pdf_file,
        output_dir="output_nemotron_results",
        dpi=250,
        merge_level="word",
        skip_relational=True,
    )