import os

from dots_ocr.parser import DotsOCRParser

MODEL_NAME = "Vishva007/dots.mocr-W4A16-AutoRound-GPTQ"
PDF_PATH = r"Docs/pan2.pdf"
OUTPUT_DIR = "./Output-Quant/example-4"
IP="213.173.109.172"
PORT=34923

def main():
    # 2. Instantiate DotsOCRParser
    parser = DotsOCRParser(
        ip=IP,  
        port=PORT,
        model_name=MODEL_NAME,
        temperature=0.1,
        top_p=0.9,
        dpi=250,                        # Rendering resolution for PDF page conversion
        max_completion_tokens=11384,
        num_thread=10,                   # Parallel thread count for processing multi-page PDFs
        output_dir=OUTPUT_DIR,
        use_hf=False                    # False uses your remote vLLM endpoint
    )

    print(f"Starting inference on {PDF_PATH} via RunPod vLLM...")

    # 3. Parse PDF
    # Available prompt modes:
    # - "prompt_layout_all_en": Detects layout boxes + full content OCR (Default)
    # - "prompt_layout_only_en": Bounding boxes and layout category detection only
    # - "prompt_ocr": Plain OCR extraction
    results = parser.parse_file(
        input_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        prompt_mode="prompt_layout_all_en"
    )

    print("\nProcessing complete!")
    print(f"Artifacts saved in: {os.path.abspath(OUTPUT_DIR)}")

    # 4. Access the parsed results
    for page in results:
        print(f"\n--- Page {page['page_no'] + 1} ---")
        print(f"Annotated Image : {page.get('layout_image_path')}")
        print(f"Extracted BBoxes: {page.get('layout_info_path')}")
        print(f"Markdown Text   : {page.get('md_content_path')}")


if __name__ == "__main__":
    main()