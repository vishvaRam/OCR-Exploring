import json
import os
import time
from typing import Any, Optional

import pymupdf  # PyMuPDF
import requests
from dots_ocr.parser import DotsOCRParser
from openai import OpenAI
from pydantic import BaseModel, Field

# ==============================================================================
# 1. Configuration
# ==============================================================================
DOTS_OCR_IP = "0.0.0.0"
DOTS_OCR_PORT = "8910"
DOTS_MODEL_NAME = "LLM"

NEMOTRON_OCR_URL = "http://0.0.0.0:8000/v1/ocr/process"
VLLM_BASE_URL = "http://0.0.0.0:7413/v1"
VLLM_MODEL_NAME = "LLM"

PDF_PATH = r"KVB_Data\velu invoice_004169_01082026_H16M29S30.pdf"
OUTPUT_DIR = "./output_results"


# ==============================================================================
# 2. Grounding & Extraction Schema
# ==============================================================================
class GroundedField(BaseModel):
    value: str = Field(description="The extracted value exactly as matched")
    page_number: int = Field(
        description="1-indexed page number where this field is located"
    )
    text_ids: list[str] = Field(
        description="Exact OCR prediction IDs grounding this value (e.g. ['p1_item_2', 'p1_item_3'])"
    )


class KFSLoanExtraction(BaseModel):
    borrower_name: GroundedField | None = Field(
        default=None, description="Borrower's full name"
    )
    loan_account_or_proposal_no: Optional[GroundedField] = Field(
        default=None, description="Proposal or Account number"
    )
    loan_type: Optional[GroundedField] = Field(
        default=None, description="Type of loan (e.g. VL4W)"
    )
    sanctioned_amount: Optional[GroundedField] = Field(
        default=None, description="Sanctioned principal loan amount"
    )
    loan_term: Optional[GroundedField] = Field(
        default=None, description="Loan term/tenor"
    )
    repayment_commencement_date: Optional[GroundedField] = Field(
        default=None, description="Repayment commencement date"
    )
    epi_installment_amount: Optional[GroundedField] = Field(
        default=None, description="Monthly instalment / EMI amount"
    )
    number_of_epis: Optional[GroundedField] = Field(
        default=None, description="Total number of instalments/EPIs"
    )
    interest_rate: Optional[GroundedField] = Field(
        default=None, description="Interest rate percentage and type (fixed/floating)"
    )
    annual_percentage_rate: Optional[GroundedField] = Field(
        default=None, description="APR percentage"
    )
    processing_fees: Optional[GroundedField] = Field(
        default=None, description="Processing fees charged"
    )
    net_disbursed_amount: Optional[GroundedField] = Field(
        default=None, description="Net disbursed loan amount"
    )


class VehicleRCExtraction(BaseModel):
    vehicle_registration_number: Optional[GroundedField] = Field(
        default=None,
        description="Vehicle registration number or license plate number (e.g., TN01AB1234)",
    )
    customer_name: Optional[GroundedField] = Field(
        default=None,
        description="Full name of the registered owner / customer",
    )
    make_and_model: Optional[GroundedField] = Field(
        default=None,
        description="Manufacturer make and specific vehicle model (e.g., MARUTI SUZUKI SWIFT VXI)",
    )
    chassis_number: Optional[GroundedField] = Field(
        default=None,
        description="Chassis number or VIN (Vehicle Identification Number)",
    )
    engine_number: Optional[GroundedField] = Field(
        default=None,
        description="Engine or motor number of the vehicle",
    )
    hypothecation_clause: Optional[GroundedField] = Field(
        default=None,
        description="Hypothecation, lease, or hire-purchase details along with the financier/bank name (e.g., HPA with KVB)",
    )
    expiry_date: Optional[GroundedField] = Field(
        default=None,
        description="Registration validity date, fitness valid until, or RC expiry date",
    )
    insured_amount: Optional[GroundedField] = Field(
        default=None,
        description="Insured Declared Value (IDV) or policy coverage amount if mentioned on the RC endorsement",
    )
    vehicle_cost: Optional[GroundedField] = Field(
        default=None,
        description="Purchase cost, ex-showroom price, or invoice value of the vehicle",
    )


# ==============================================================================
# 3. DotsOCR Markdown Extraction
# ==============================================================================
def extract_markdown(pdf_path: str, output_dir: str) -> tuple[str, str]:
    """Generates markdown layout via DotsOCR and writes a combined .md file."""
    parser = DotsOCRParser(
        ip=DOTS_OCR_IP,
        port=DOTS_OCR_PORT,
        model_name=DOTS_MODEL_NAME,
        temperature=0.1,
        top_p=0.9,
        dpi=200,
        max_completion_tokens=11384,
        num_thread=6,
        output_dir=output_dir,
        use_hf=False,
    )

    results = parser.parse_file(
        input_path=pdf_path,
        output_dir=output_dir,
        prompt_mode="prompt_markdown",
    )

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    combined_md_path = os.path.join(output_dir, f"{base_name}.md")
    all_markdown_content = []

    sorted_results = sorted(results, key=lambda x: x.get("page_no", 0))
    for page in sorted_results:
        page_num = page.get("page_no", 0) + 1
        md_file = page.get("md_content_path")

        page_text = ""
        if md_file and os.path.exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                page_text = f.read()
        elif "content" in page:
            page_text = page.get("content", "")
        elif "raw_output" in page:
            page_text = page.get("raw_output", "")

        all_markdown_content.append(
            f"<!-- Page {page_num} -->\n\n{page_text.strip()}\n"
        )

    full_markdown = "\n\n---\n\n".join(all_markdown_content)
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    return full_markdown, combined_md_path


# ==============================================================================
# 4. Nemotron OCR Text & ID Extraction
# ==============================================================================
def extract_nemotron_ocr(pdf_path: str, output_dir: str) -> tuple[dict[str, Any], str]:
    """Posts PDF to the Nemotron OCR endpoint and saves the raw JSON response."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    ocr_json_path = os.path.join(output_dir, f"{base_name}_ocr.json")

    with open(pdf_path, "rb") as f:
        response = requests.post(
            NEMOTRON_OCR_URL,
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
            data={
                "dpi": 200,
                "merge_level": "word",
                "skip_relational": True,
                "batch_size": 2,
            },
            timeout=300,
        )

    response.raise_for_status()
    ocr_data = response.json()

    clean_ocr_data = {k: v for k, v in ocr_data.items() if k != "annotated_pdf_base64"}
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_ocr_data, f, indent=2)

    return clean_ocr_data, ocr_json_path


# ==============================================================================
# 5. LLM Structured Grounded Extraction
# ==============================================================================
def build_grounding_catalog(ocr_data: dict[str, Any]) -> str:
    """Formats OCR tokens and their IDs per page for the LLM prompt."""
    catalog_lines = []
    for page in ocr_data.get("pages", []):
        page_num = page.get("page_number", 1)
        catalog_lines.append(f"=== PAGE {page_num} OCR ELEMENTS ===")
        for item in page.get("predictions", []):
            item_id = item.get("id")
            text = item.get("text", "").strip()
            if text:
                catalog_lines.append(f"[{item_id}]: {text}")
        catalog_lines.append("")
    return "\n".join(catalog_lines)


def run_llm_extraction(
    markdown_text: str, ocr_data: dict[str, Any], model_name: str, base_url: str
) -> Any:
    """Invokes vLLM with a domain-agnostic grounding prompt and returns parsed fields."""
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    ocr_catalog = build_grounding_catalog(ocr_data)

    system_prompt = """You are an advanced document extraction and visual grounding engine.

Your objective is to extract structured target fields from any document type (forms, invoices, reports, contracts, tables, or unstructured text) and ground each extracted value to its corresponding OCR token IDs.

### Grounding & Extraction Protocol:

1. Structural Cross-Referencing:
   - Use the **Document Markdown Layout** to understand layout hierarchy, section divisions, tabular grids, and key-value associations.
   - Use the **OCR Grounding Catalog** to locate the exact bounding token IDs (`p<page>_item_<index>`) that correspond to the extracted values[cite: 1].

2. Value-Only Grounding (Exclude Labels & Headers):
   - Ground only the **actual data value**, NEVER the field label, prompt, or column header.
   - *Example:* If the text is "Due Date: 12/04/2026", select the token IDs for "12/04/2026". Exclude "Due" and "Date:".
   - *Example:* For table rows, ground the data cells, not the column headers.

3. Multi-Token Continuity & Sequence:
   - When an extracted value spans multiple adjacent OCR tokens (e.g., full names, addresses, currency amounts with symbols, or alphanumeric codes), collect every constituent token ID in reading order (left-to-right, top-to-bottom).

4. Page-Scope Consistency:
   - Ensure the 1-indexed `page_number` strictly matches the page prefix of every selected token ID (e.g., elements on page 2 must exclusively reference `p2_item_*`).

5. Verbatim Accuracy & Handling Missing Data:
   - Do not hallucinate, calculate, or synthesize values that do not explicitly appear in the source text.
   - If a requested field cannot be found, is blank, or is explicitly marked as absent/not applicable, return null for that field."""

    user_content = f"""Please extract the target fields defined in the schema from the document data below.

================================================================================
DOCUMENT MARKDOWN LAYOUT
================================================================================
{markdown_text}

================================================================================
OCR GROUNDING CATALOG (TOKEN IDS & TEXT)
================================================================================
{ocr_catalog}

================================================================================
EXECUTION INSTRUCTIONS:
1. Parse the document using the Markdown layout to locate the requested fields.
2. Cross-reference the identified values with the OCR catalog to retrieve their exact token IDs.
3. Populate and return the target Pydantic schema with the extracted values, page numbers, and token IDs.
================================================================================"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    t_api_start = time.perf_counter()
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=VehicleRCExtraction,
        temperature=0.0,
    )
    api_elapsed = time.perf_counter() - t_api_start

    print("\n" + "=" * 80)
    print(f" TOTAL CONTEXT SENT TO LLM API ({model_name})")
    print("=" * 80)
    print(f"--- SYSTEM PROMPT ---\n{system_prompt}\n")
    print(f"--- USER PROMPT ---\n{user_content}")
    print("=" * 80)

    usage = completion.usage
    print("\n[API Metrics - LLM Inference]")
    print(f"  * Call Latency      : {api_elapsed:.2f} seconds")
    if usage:
        print(f"  * Prompt Tokens     : {usage.prompt_tokens:,}")
        print(f"  * Completion Tokens : {usage.completion_tokens:,}")
        print(f"  * Total Tokens      : {usage.total_tokens:,}")
    else:
        print(
            f"  * Context Characters: {len(system_prompt) + len(user_content):,} chars"
        )
    print("=" * 80 + "\n")

    return completion.choices[0].message.parsed


# ==============================================================================
# 6. PDF Annotation
# ==============================================================================
def get_prediction_rect(
    pred: dict[str, Any], page_w: float, page_h: float
) -> Optional[pymupdf.Rect]:
    """Calculates PyMuPDF Rect from normalized coordinates, quad, or bbox."""
    if pred.get("quad"):
        quad = pred["quad"]
        xs = [pt[0] for pt in quad]
        ys = [pt[1] for pt in quad]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
    elif all(k in pred for k in ("left", "right", "upper", "lower")):
        x0, x1 = sorted([pred["left"], pred["right"]])
        y0, y1 = sorted([pred["upper"], pred["lower"]])
    elif "bbox" in pred:
        x0, y0, x1, y1 = pred["bbox"]
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
    else:
        return None

    if max(x0, x1, y0, y1) <= 1.0:
        rx0, rx1 = x0 * page_w, x1 * page_w
        ry0, ry1 = y0 * page_h, y1 * page_h
    else:
        rx0, rx1, ry0, ry1 = x0, x1, y0, y1

    rx0, rx1 = max(0.0, min(rx0, page_w)), max(0.0, min(rx1, page_w))
    ry0, ry1 = max(0.0, min(ry0, page_h)), max(0.0, min(ry1, page_h))

    if (rx1 - rx0) < 1.0 or (ry1 - ry0) < 1.0:
        return None

    return pymupdf.Rect(rx0, ry0, rx1, ry1)


def annotate_grounded_boxes(
    original_pdf_path: str,
    ocr_data: dict[str, Any],
    extraction: KFSLoanExtraction,
    output_pdf_path: str,
) -> None:
    """Overlays vector rectangles only on fields matched by the LLM."""
    doc = pymupdf.open(original_pdf_path)

    id_to_pred = {}
    for page in ocr_data.get("pages", []):
        p_num = page.get("page_number", 1)
        for pred in page.get("predictions", []):
            id_to_pred[(p_num, pred["id"])] = pred

    target_ids_by_page: dict[int, list[tuple[str, str, str]]] = {}
    for field_name, grounded_field in extraction.model_dump().items():
        if not grounded_field:
            continue
        p_num = grounded_field.get("page_number")
        val = grounded_field.get("value", "")
        for tid in grounded_field.get("text_ids", []):
            target_ids_by_page.setdefault(p_num, []).append((tid, field_name, val))

    highlight_color = (0.85, 0.12, 0.12)

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        if page_num not in target_ids_by_page:
            continue

        page = doc[page_idx]
        page_w = page.rect.width
        page_h = page.rect.height

        for tid, field_name, val in target_ids_by_page[page_num]:
            pred = id_to_pred.get((page_num, tid))
            if not pred:
                continue

            rect = get_prediction_rect(pred, page_w, page_h)
            if not rect:
                continue

            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=highlight_color)
            annot.set_border(width=1.5)
            annot.set_info(title="Extracted Field", content=f"{field_name}: {val}")
            annot.update()

    doc.save(output_pdf_path, garbage=3, deflate=True)
    doc.close()


# ==============================================================================
# 7. Orchestrator with Step Timers
# ==============================================================================
def process_document(pdf_path: str, output_dir: str = OUTPUT_DIR):
    total_pipeline_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print("\n" + "#" * 60)
    print(f" PIPELINE EXECUTION START: {os.path.basename(pdf_path)}")
    print("#" * 60)

    # Step 1: DotsOCR Markdown
    t0 = time.perf_counter()
    print("\n[1/4] Generating Markdown via DotsOCR...")
    markdown_text, md_path = extract_markdown(pdf_path, output_dir)
    step1_time = time.perf_counter() - t0
    print(f"      Completed in: {step1_time:.2f}s | Saved -> {md_path}")

    # Step 2: Nemotron OCR
    t1 = time.perf_counter()
    print(f"\n[2/4] Calling Nemotron OCR API ({NEMOTRON_OCR_URL})...")
    ocr_data, ocr_json_path = extract_nemotron_ocr(pdf_path, output_dir)
    step2_time = time.perf_counter() - t1
    print(f"      Completed in: {step2_time:.2f}s | Saved -> {ocr_json_path}")

    # Step 3: LLM Extraction & Context Display
    t2 = time.perf_counter()
    print(f"\n[3/4] Running Structured Grounding via vLLM ({VLLM_MODEL_NAME})...")
    extracted_data = run_llm_extraction(
        markdown_text=markdown_text,
        ocr_data=ocr_data,
        model_name=VLLM_MODEL_NAME,
        base_url=VLLM_BASE_URL,
    )
    extracted_json_path = os.path.join(output_dir, f"{base_name}_extracted.json")
    with open(extracted_json_path, "w", encoding="utf-8") as f:
        f.write(extracted_data.model_dump_json(indent=2))
    step3_time = time.perf_counter() - t2
    print(
        f"      Step 3 Completed in: {step3_time:.2f}s | Saved -> {extracted_json_path}"
    )

    # Step 4: Annotation
    t3 = time.perf_counter()
    print("\n[4/4] Annotating Target Vector Bounding Boxes on PDF...")
    annotated_pdf_path = os.path.join(output_dir, f"{base_name}_grounded.pdf")
    annotate_grounded_boxes(
        original_pdf_path=pdf_path,
        ocr_data=ocr_data,
        extraction=extracted_data,
        output_pdf_path=annotated_pdf_path,
    )
    step4_time = time.perf_counter() - t3
    print(f"      Completed in: {step4_time:.2f}s | Saved -> {annotated_pdf_path}")

    # Total Benchmark Summary
    total_elapsed = time.perf_counter() - total_pipeline_start
    print("\n" + "=" * 60)
    print(" EXECUTION TIME SUMMARY")
    print("=" * 60)
    print(f"  * 1. DotsOCR Markdown Extraction : {step1_time:6.2f}s")
    print(f"  * 2. Nemotron OCR API Call       : {step2_time:6.2f}s")
    print(f"  * 3. vLLM Grounded Extraction    : {step3_time:6.2f}s")
    print(f"  * 4. PyMuPDF Vector Annotation   : {step4_time:6.2f}s")
    print("-" * 60)
    print(f"  * TOTAL PIPELINE TIME            : {total_elapsed:6.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    process_document(PDF_PATH, OUTPUT_DIR)
