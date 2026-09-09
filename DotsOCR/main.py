import os

from dots_ocr.parser import DotsOCRParser

MODEL_NAME = "LLM"                 # Model name passed to `vllm serve --served-model-name`
PDF_PATH = r"KVB_Data\Document_Per_Account_Vechicle_Loan\1259775000000530\KFS_Statement\KFS S_019321_05062026_H12M54S36.pdf"
OUTPUT_DIR = "./output_results"      # Directory where marked images & JSONs will be saved


def main():
    # 2. Instantiate DotsOCRParser
    parser = DotsOCRParser(
        ip="192.9.200.29",  
        port="8910",
        model_name=MODEL_NAME,
        temperature=0.1,
        top_p=0.9,
        dpi=200,                        # Rendering resolution for PDF page conversion
        max_completion_tokens=11384,
        num_thread=6,                   # Parallel thread count for processing multi-page PDFs
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