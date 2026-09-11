import os

from dots_ocr.parser import DotsOCRParser

MODEL_NAME = "LLM"
PDF_PATH = r"input.pdf"
OUTPUT_DIR = "./output_results"


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Instantiate DotsOCRParser
    parser = DotsOCRParser(
        ip="0.0.0.0",
        port="8910",
        model_name=MODEL_NAME,
        temperature=0.1,
        top_p=0.9,
        dpi=200,
        max_completion_tokens=11384,
        num_thread=6,
        output_dir=OUTPUT_DIR,
        use_hf=False,
    )

    print(f"Starting inference on {PDF_PATH} via vLLM...")

    # 2. Parse PDF
    # Use "prompt_layout_all_en" if you want dots.ocr's automatic layout-to-markdown engine
    # Or use your custom "prompt_markdown"
    PROMPT_MODE = "prompt_markdown"  # or "prompt_markdown"

    results = parser.parse_file(
        input_path=PDF_PATH,
        output_dir=OUTPUT_DIR,
        prompt_mode=PROMPT_MODE,
    )

    print("\nProcessing complete!")

    # 3. Aggregate markdown from all pages and save to a unified .md file
    base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    combined_md_path = os.path.join(OUTPUT_DIR, f"{base_name}.md")

    all_markdown_content = []

    # Sort results by page number to guarantee correct reading order
    sorted_results = sorted(results, key=lambda x: x.get("page_no", 0))

    for page in sorted_results:
        page_num = page.get("page_no", 0) + 1
        md_file = page.get("md_content_path")
        
        page_text = ""
        # Case A: dots_ocr auto-generated the markdown file
        if md_file and os.path.exists(md_file):
            with open(md_file, "r", encoding="utf-8") as f:
                page_text = f.read()
        # Case B: raw markdown text is in the response dict
        elif "content" in page:
            page_text = page.get("content", "")
        elif "raw_output" in page:
            page_text = page.get("raw_output", "")

        all_markdown_content.append(f"<!-- Page {page_num} -->\n\n{page_text.strip()}\n")

    # Write out the combined Markdown file
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_markdown_content))

    print(f"\nFinal combined Markdown saved to: {os.path.abspath(combined_md_path)}")


if __name__ == "__main__":
    main()