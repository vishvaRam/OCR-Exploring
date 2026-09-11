import asyncio
import base64
import gc
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

import pymupdf  # PyMuPDF
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. Logging & Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nemotron-ocr-api")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# -----------------------------------------------------------------------------
# 2. Application Lifespan & Global State
# -----------------------------------------------------------------------------
class MLState:
    ocr_engine: Optional[NemotronOCRV2] = None
    inference_lock: asyncio.Lock = asyncio.Lock()


ml_state = MLState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once into GPU memory on startup and tear down gracefully."""
    logger.info("Initializing NemotronOCRV2 engine into GPU memory...")
    try:
        ml_state.ocr_engine = NemotronOCRV2(skip_relational=True)
        logger.info("NemotronOCRV2 engine loaded successfully.")
    except Exception as exc:
        logger.critical(f"Failed to load NemotronOCRV2: {exc}", exc_info=True)
        raise exc

    yield

    logger.info("Tearing down ML context...")
    ml_state.ocr_engine = None


app = FastAPI(
    title="NVIDIA Nemotron OCR Service",
    version="2.0.0",
    description="High-performance OCR pipeline powered by Nemotron-OCR-v2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# 3. Schemas
# -----------------------------------------------------------------------------
class PageOCRResult(BaseModel):
    page_number: int
    text_count: int
    predictions: List[Dict[str, Any]]


class OCRResponse(BaseModel):
    filename: str
    total_pages: int
    pages: List[PageOCRResult]
    annotated_pdf_base64: str = Field(
        ...,
        description="Base64-encoded PDF containing non-destructive vector bounding boxes",
    )


# -----------------------------------------------------------------------------
# 4. PyMuPDF Direct Vector Annotation Utilities
# -----------------------------------------------------------------------------
def serialize_prediction_item(item: Any, text_id: str) -> Dict[str, Any]:
    """Ensures prediction objects become dicts and injects the unique ID as the first key."""
    if hasattr(item, "__dict__"):
        data = dict(item.__dict__)
    elif isinstance(item, dict):
        data = dict(item)
    else:
        data = {"raw_value": str(item)}

    return {"id": text_id, **data}


def annotate_page_with_pymupdf(
    page: pymupdf.Page,
    predictions: list,
    img_width: int,
    img_height: int,
) -> None:
    """Draws clean vector bounding box borders with no icons, fills, or text popups."""
    page_w = page.rect.width
    page_h = page.rect.height

    scale_x = page_w / float(img_width)
    scale_y = page_h / float(img_height)

    box_color = (0.0, 0.85, 0.3)  # Bright green (#00D94D)

    for item in predictions:
        if item.get("quad"):
            quad = item["quad"]
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
        elif all(k in item for k in ("left", "right", "upper", "lower")):
            left, right, top, bottom = (
                item["left"],
                item["right"],
                item["upper"],
                item["lower"],
            )
        elif "bbox" in item:
            left, top, right, bottom = item["bbox"]
        else:
            continue

        if max(left, right, top, bottom) <= 1.0:
            x0 = left * page_w
            x1 = right * page_w
            y0 = top * page_h
            y1 = bottom * page_h
        else:
            x0 = left * scale_x
            x1 = right * scale_x
            y0 = top * scale_y
            y1 = bottom * scale_y

        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        x0 = max(0.0, min(x0, page_w))
        x1 = max(0.0, min(x1, page_w))
        y0 = max(0.0, min(y0, page_h))
        y1 = max(0.0, min(y1, page_h))

        if (x1 - x0) < 1.0 or (y1 - y0) < 1.0:
            continue

        box_rect = pymupdf.Rect(x0, y0, x1, y1)

        rect_annot = page.add_rect_annot(box_rect)
        rect_annot.set_colors(stroke=box_color)
        rect_annot.set_border(width=1.0)
        rect_annot.update()


def run_pipeline_sync(
    pdf_path: str,
    dpi: int,
    merge_level: str,
    skip_relational: bool,
    batch_size: int = 2,
) -> tuple[List[PageOCRResult], bytes]:
    """Processes PDF pages in batches to cap peak VRAM, flushing the CUDA allocator between batches."""
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    page_results: List[PageOCRResult] = []

    try:
        with tempfile.TemporaryDirectory() as temp_render_dir:
            # Process in chunks of 'batch_size' pages
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"Processing PDF page batch: {batch_start + 1} to {batch_end} of {total_pages}")

                for page_idx in range(batch_start, batch_end):
                    page_num = page_idx + 1
                    page = doc[page_idx]

                    # Render page to disk for Nemotron OCR inference
                    zoom = dpi / 72.0
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                    img_w, img_h = pix.width, pix.height
                    temp_img_path = os.path.join(temp_render_dir, f"page_{page_num}.png")
                    pix.save(temp_img_path)

                    del pix  # Drop host pixmap immediately

                    # Model inference inside pure inference context
                    with torch.inference_mode():
                        raw_predictions = ml_state.ocr_engine(
                            temp_img_path,
                            merge_level=merge_level,
                        )

                    serialized_preds = [
                        serialize_prediction_item(pred, text_id=f"p{page_num}_item_{idx}")
                        for idx, pred in enumerate(raw_predictions)
                    ]

                    annotate_page_with_pymupdf(
                        page=page,
                        predictions=serialized_preds,
                        img_width=img_w,
                        img_height=img_h,
                    )

                    page_results.append(
                        PageOCRResult(
                            page_number=page_num,
                            text_count=len(serialized_preds),
                            predictions=serialized_preds,
                        )
                    )

                    if os.path.exists(temp_img_path):
                        os.unlink(temp_img_path)

                # Evict GPU cache and trigger Python GC between batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        annotated_pdf_bytes = doc.tobytes(garbage=3, deflate=True)
        return page_results, annotated_pdf_bytes

    finally:
        doc.close()
        del doc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# -----------------------------------------------------------------------------
# 5. API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Liveness & readiness probe."""
    if ml_state.ocr_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized.",
        )
    return {"status": "healthy", "gpu_active": True}


@app.post(
    "/v1/ocr/process",
    response_model=OCRResponse,
    summary="Process PDF with Nemotron OCR",
    description="Runs OCR in GPU-safe batches, assigns unique text IDs, and returns direct vector overlays.",
)
async def process_pdf_endpoint(
    file: UploadFile = File(..., description="Target PDF document"),
    dpi: int = Form(200, ge=72, le=400, description="DPI for model rasterization"),
    merge_level: Literal["word", "line", "paragraph"] = Form(
        "word", description="Merge level for detected bounding boxes"
    ),
    skip_relational: bool = Form(
        True, description="Bypass table and entity-relational linking for speed"
    ),
    batch_size: int = Form(
        2, ge=1, le=10, description="Number of pages to process before flushing GPU memory cache"
    ),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are accepted.",
        )

    file_size = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
        tmp_file_path = tmp_input.name
        try:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB.",
                    )
                tmp_input.write(chunk)
        except Exception:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            raise

    try:
        async with ml_state.inference_lock:
            loop = asyncio.get_running_loop()
            page_results, annotated_pdf_bytes = await loop.run_in_executor(
                None,
                run_pipeline_sync,
                tmp_file_path,
                dpi,
                merge_level,
                skip_relational,
                batch_size,
            )

        encoded_pdf = base64.b64encode(annotated_pdf_bytes).decode("utf-8")

        return OCRResponse(
            filename=file.filename,
            total_pages=len(page_results),
            pages=page_results,
            annotated_pdf_base64=encoded_pdf,
        )

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Error executing OCR inference: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR execution failed: {str(err)}",
        )
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )