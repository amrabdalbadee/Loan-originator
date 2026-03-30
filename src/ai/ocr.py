from typing import Dict, Any, List, Optional
import os
import torch
from PIL import Image
from src.core.config import settings
from src.ai import model_handler_chandra2

class OCREngine:
    def __init__(self, model_name: str = "Chandra2"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None

    def _load_model(self):
        """Lazy load the model if not already loaded."""
        if self.model is None:
            if self.model_name == "Chandra2":
                self.model, self.processor, self.device, _, _ = model_handler_chandra2.load_model()
            else:
                # Placeholder for other models like Qwen2-VL
                self.device = model_handler_chandra2.get_device()

    async def extract_text(self, document_image_path: str) -> Dict[str, Any]:
        """
        Extract text from document image using the selected model.
        Returns a dictionary with extracted fields (if parsed) and raw output.
        """
        if not os.path.exists(document_image_path):
            return {"status": "error", "message": f"File not found: {document_image_path}"}

        self._load_model()
        
        try:
            image = Image.open(document_image_path)
            
            if self.model_name == "Chandra2":
                # Chandra2 'ocr' mode returns parsed markdown/text
                raw_result = model_handler_chandra2.extract_table_from_image(
                    image, self.model, self.processor, self.device, prompt="USE_NATIVE"
                )
                
                # In a real scenario, we'd have a parser here to extract fields.
                # For now, we return the raw text and a status.
                return {
                    "raw_text": raw_result,
                    "fields": {
                        # Placeholder extraction logic
                        "full_name": {"value": "Extracted from OCR", "confidence": 0.90},
                    },
                    "status": "success" if "[ERROR]" not in raw_result else "error",
                    "model_version": "Chandra-OCR-2"
                }
            
            # Default placeholder result
            return {
              "fields": {
                  "full_name": {"value": "John Doe", "confidence": 0.95},
                  "nid_number": {"value": "12345678901234", "confidence": 0.98},
                  "dob": {"value": "1990-01-01", "confidence": 0.99},
                  "expiry_date": {"value": "2030-01-01", "confidence": 0.99},
              },
              "status": "success",
              "model_version": "v1.0.0"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def validate_confidence(self, extraction_result: Dict[str, Any]) -> List[str]:
        """
        FR-0.1: Flag fields with OCR confidence < settings.OCR_AUTO_ACCEPT_THRESHOLD.
        """
        low_confidence_fields = []
        for field, data in extraction_result.get("fields", {}).items():
            if isinstance(data, dict) and "confidence" in data:
                if data["confidence"] < settings.OCR_AUTO_ACCEPT_THRESHOLD:
                    low_confidence_fields.append(field)
        return low_confidence_fields

ocr_engine = OCREngine()
