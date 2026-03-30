import asyncio
import sys
import os
from pathlib import Path

# Add project root to sys.path to resolve 'src' imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from src.ai.ocr import OCREngine

async def try_chandra2(image_path: str):
    """
    Sample script to run Chandra OCR 2 on a specified document image.
    """
    print(f"--- Chandra OCR 2 Test ---")
    print(f"Target Document: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"[ERROR] Specified image path does not exist.")
        return

    # Initialize the OCR engine (defaults to Chandra2)
    engine = OCREngine(model_name="Chandra2")
    
    print("Loading model and performing extraction (this may take some time)...")
    try:
        # Perform extraction
        result = await engine.extract_text(image_path)
        
        print("\n--- Extraction Results ---")
        print(f"Status: {result.get('status')}")
        print(f"Model: {result.get('model_version')}")
        
        if result.get("status") == "success":
            print("\nRAW OCR OUTPUT (Markdown/HTML):")
            print("-" * 30)
            print(result.get("raw_text"))
            print("-" * 30)
            
            # Fields should be empty according to the current logic
            if result.get("fields"):
                print("\nEXTRACTED FIELDS:")
                print(json.dumps(result.get("fields"), indent=2, ensure_ascii=False))
            else:
                print("\n[NOTE] No structured fields extracted yet; raw output provided above.")
        else:
            print(f"[ERROR] Extraction failed: {result.get('message') or result.get('raw_text')}")
            
    except Exception as e:
        print(f"[FATAL] An error occurred during the test: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/try_chandra2.py <path_to_document_image>")
        sys.exit(1)
        
    image_to_test = sys.argv[1]
    
    # Run the async test
    asyncio.run(try_chandra2(image_to_test))
