import io
import itertools
import json
import re
import time
from pathlib import Path
from typing import Literal

import pymupdf  # PyMuPDF
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem, TextItem
from docling_core.types.doc.base import CoordOrigin
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Schemas
# ---------------------------------------------------------------------------
class OCRElement(BaseModel):
    id: int
    page_idx: int = Field(..., description="0-indexed page number")
    text: str
    granularity: str
    pdf_bbox: list[float] = Field(
        ..., description="[x0, y0, x1, y1] in standard PDF point coordinates"
    )


class OCRResult(BaseModel):
    filename: str
    total_pages: int
    granularity: str
    total_boxes: int
    processing_time_sec: float
    pure_text: str
    ocr_elements: list[OCRElement]


# ---------------------------------------------------------------------------
# 2. Docling Converter Initialization
# ---------------------------------------------------------------------------
def init_docling_converter() -> DocumentConverter:
    accelerator_options = AcceleratorOptions(
        num_threads=16,
        device=AcceleratorDevice.CUDA,
        cuda_use_flash_attention2=True,
    )

    # Note: RapidOCR requires .onnx model weights.
    ocr_options = RapidOcrOptions(
        backend="onnxruntime",
        scale=3.0,
        text_score=0.4,
        use_cls=False,
        det_model_path="Models/PP-OCRv6_medium_det_onnx/inference.onnx",
        rec_model_path="Models/PP-OCRv6_medium_rec_onnx/inference.onnx",
        rec_keys_path="Models/PP-OCRv6_medium_rec_onnx/inference.yml",
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_table_structure = True  # Required to extract table cells
    pipeline_options.ocr_batch_size = 8

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


_WORD_REGEX = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------------------------
def _split_into_word_bboxes(
    line_text: str,
    line_bbox: list[float],
) -> list[tuple[str, list[float]]]:
    x0, y0, x1, y1 = line_bbox
    line_len = len(line_text)
    line_width = x1 - x0

    if line_len == 0 or line_width <= 0:
        return [(line_text, line_bbox)]

    words: list[tuple[str, list[float]]] = []
    ratio = line_width / line_len

    for match in _WORD_REGEX.finditer(line_text):
        word_x0 = x0 + match.start() * ratio
        word_x1 = x0 + match.end() * ratio
        words.append(
            (
                match.group(),
                [
                    round(word_x0, 2),
                    round(y0, 2),
                    round(word_x1, 2),
                    round(y1, 2),
                ],
            )
        )
    return words


# ---------------------------------------------------------------------------
# 4. Processing Core
# ---------------------------------------------------------------------------
def process_pdf(
    pdf_path: str | Path,
    converter: DocumentConverter,
    granularity: Literal["element", "line", "word"] = "line",
    generate_annotated_pdf: bool = True,
) -> tuple[OCRResult, bytes | None]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    start_time = time.perf_counter()
    pdf_bytes = pdf_path.read_bytes()

    doc_stream = DocumentStream(name=pdf_path.name, stream=io.BytesIO(pdf_bytes))
    conv_result = converter.convert(doc_stream)
    dl_doc = conv_result.document
    pure_text = dl_doc.export_to_text()

    pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(pdf_doc)

    ocr_elements: list[OCRElement] = []
    element_id = 0

    # Chain body elements and furniture items (captures running headers/footers)
    furniture_items = (
        dl_doc.iterate_items(root=dl_doc.furniture)
        if (getattr(dl_doc, "furniture", None) and dl_doc.furniture.children)
        else []
    )
    all_items = itertools.chain(dl_doc.iterate_items(), furniture_items)

    for item, _ in all_items:
        # -------------------------------------------------------------------
        # Case A: Standard Text Items (Paragraphs, Headings, Headers, Labels)
        # -------------------------------------------------------------------
        if isinstance(item, TextItem):
            if not (item.text.strip() and item.prov):
                continue

            if granularity == "element":
                prov = item.prov[0]
                page_idx = prov.page_no - 1
                if page_idx >= total_pages:
                    continue

                page_rect = pdf_doc[page_idx].rect
                dl_page = dl_doc.pages.get(prov.page_no)
                dl_page_w = dl_page.size.width if dl_page else page_rect.width
                dl_page_h = dl_page.size.height if dl_page else page_rect.height

                scale_x = page_rect.width / dl_page_w
                scale_y = page_rect.height / dl_page_h

                bbox = prov.bbox
                if bbox.coord_origin != CoordOrigin.TOPLEFT:
                    bbox = bbox.to_top_left_origin(page_height=dl_page_h)

                ocr_elements.append(
                    OCRElement(
                        id=element_id,
                        page_idx=page_idx,
                        text=item.text.strip(),
                        granularity="element",
                        pdf_bbox=[
                            round(bbox.l * scale_x, 2),
                            round(bbox.t * scale_y, 2),
                            round(bbox.r * scale_x, 2),
                            round(bbox.b * scale_y, 2),
                        ],
                    )
                )
                element_id += 1
                continue

            # Line or Word level
            for prov in item.prov:
                page_idx = prov.page_no - 1
                if page_idx >= total_pages:
                    continue

                page_rect = pdf_doc[page_idx].rect
                dl_page = dl_doc.pages.get(prov.page_no)
                dl_page_w = dl_page.size.width if dl_page else page_rect.width
                dl_page_h = dl_page.size.height if dl_page else page_rect.height

                scale_x = page_rect.width / dl_page_w
                scale_y = page_rect.height / dl_page_h

                charspan = getattr(prov, "charspan", None)
                if charspan:
                    start = (
                        charspan[0]
                        if isinstance(charspan, (list, tuple))
                        else getattr(charspan, "start", 0)
                    )
                    end = (
                        charspan[1]
                        if isinstance(charspan, (list, tuple))
                        else getattr(charspan, "end", len(item.text))
                    )
                    line_text = item.text[start:end].strip()
                else:
                    line_text = item.text.strip()

                if not line_text:
                    continue

                bbox = prov.bbox
                if bbox.coord_origin != CoordOrigin.TOPLEFT:
                    bbox = bbox.to_top_left_origin(page_height=dl_page_h)

                line_bbox = [
                    round(bbox.l * scale_x, 2),
                    round(bbox.t * scale_y, 2),
                    round(bbox.r * scale_x, 2),
                    round(bbox.b * scale_y, 2),
                ]

                if granularity == "word":
                    for w_text, w_bbox in _split_into_word_bboxes(line_text, line_bbox):
                        ocr_elements.append(
                            OCRElement(
                                id=element_id,
                                page_idx=page_idx,
                                text=w_text,
                                granularity="word",
                                pdf_bbox=w_bbox,
                            )
                        )
                        element_id += 1
                else:
                    ocr_elements.append(
                        OCRElement(
                            id=element_id,
                            page_idx=page_idx,
                            text=line_text,
                            granularity="line",
                            pdf_bbox=line_bbox,
                        )
                    )
                    element_id += 1

        # -------------------------------------------------------------------
        # Case B: Table Items (Captures tables and structured cells)
        # -------------------------------------------------------------------
        elif isinstance(item, TableItem):
            if not item.prov:
                continue

            prov = item.prov[0]
            page_idx = prov.page_no - 1
            if page_idx >= total_pages:
                continue

            page_rect = pdf_doc[page_idx].rect
            dl_page = dl_doc.pages.get(prov.page_no)
            dl_page_w = dl_page.size.width if dl_page else page_rect.width
            dl_page_h = dl_page.size.height if dl_page else page_rect.height

            scale_x = page_rect.width / dl_page_w
            scale_y = page_rect.height / dl_page_h

            if item.data and item.data.table_cells:
                for cell in item.data.table_cells:
                    cell_text = cell.text.strip() if cell.text else ""
                    if not cell_text:
                        continue

                    # Fall back to whole table bbox if cell bbox is omitted
                    bbox = cell.bbox if cell.bbox else prov.bbox
                    if bbox.coord_origin != CoordOrigin.TOPLEFT:
                        bbox = bbox.to_top_left_origin(page_height=dl_page_h)

                    cell_bbox = [
                        round(bbox.l * scale_x, 2),
                        round(bbox.t * scale_y, 2),
                        round(bbox.r * scale_x, 2),
                        round(bbox.b * scale_y, 2),
                    ]

                    if granularity == "word":
                        for w_text, w_bbox in _split_into_word_bboxes(
                            cell_text, cell_bbox
                        ):
                            ocr_elements.append(
                                OCRElement(
                                    id=element_id,
                                    page_idx=page_idx,
                                    text=w_text,
                                    granularity="word",
                                    pdf_bbox=w_bbox,
                                )
                            )
                            element_id += 1
                    else:
                        ocr_elements.append(
                            OCRElement(
                                id=element_id,
                                page_idx=page_idx,
                                text=cell_text,
                                granularity="line"
                                if granularity == "line"
                                else "element",
                                pdf_bbox=cell_bbox,
                            )
                        )
                        element_id += 1

    # Render PDF highlights
    annotated_pdf_bytes = None
    if generate_annotated_pdf:
        for elem in ocr_elements:
            pdf_page = pdf_doc[elem.page_idx]
            rect = pymupdf.Rect(*elem.pdf_bbox)
            if (
                rect.is_valid
                and not rect.is_empty
                and rect.width > 0
                and rect.height > 0
            ):
                annot = pdf_page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.85, 0.1))
                annot.set_info(content=f"[{elem.id}] {elem.text}")
                annot.update()

        out_buf = io.BytesIO()
        pdf_doc.save(out_buf, garbage=3, deflate=True)
        annotated_pdf_bytes = out_buf.getvalue()

    pdf_doc.close()
    elapsed = round(time.perf_counter() - start_time, 3)

    result = OCRResult(
        filename=pdf_path.name,
        total_pages=total_pages,
        granularity=granularity,
        total_boxes=len(ocr_elements),
        processing_time_sec=elapsed,
        pure_text=pure_text,
        ocr_elements=ocr_elements,
    )

    return result, annotated_pdf_bytes


# ---------------------------------------------------------------------------
# 5. Execution Entry Point (Static Configuration)
# ---------------------------------------------------------------------------
PDF_PATH = (
    r"Docs/VAHAN INS INVOICE_014002_16062026_H9M7S40.pdf"  # Path to your input PDF
)
GRANULARITY: Literal["element", "line", "word"] = "line"  # "element", "line", or "word"
OUTPUT_DIR = "./output"
GENERATE_ANNOTATED_PDF = True


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Initializing Docling Pipeline...")
    converter = init_docling_converter()

    print(f"Processing '{PDF_PATH}' with granularity='{GRANULARITY}'...")
    result, annotated_bytes = process_pdf(
        pdf_path=PDF_PATH,
        converter=converter,
        granularity=GRANULARITY,
        generate_annotated_pdf=GENERATE_ANNOTATED_PDF,
    )

    # Save JSON results
    json_path = output_path / f"{Path(PDF_PATH).stem}_ocr.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    # Save annotated PDF if requested
    if annotated_bytes:
        annotated_pdf_path = (
            output_path / f"annotated_{GRANULARITY}_{Path(PDF_PATH).name}"
        )
        annotated_pdf_path.write_bytes(annotated_bytes)
        print(f"Saved annotated PDF to: {annotated_pdf_path}")

    print(f"Saved OCR metadata to: {json_path}")
    print(f"Extracted {result.total_boxes} boxes in {result.processing_time_sec}s")


if __name__ == "__main__":
    main()
