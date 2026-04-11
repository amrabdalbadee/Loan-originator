from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import sys
import json
from PIL import Image
from datetime import datetime

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from extractor import EgyptianIDExtractor, PassportExtractor

app = FastAPI(title="Identity Extraction Service")

@app.post("/extract")
async def extract_identity(
    doc_type: str = Form(...),
    id_front: UploadFile = File(None),
    id_back: UploadFile = File(None),
    passport: UploadFile = File(None)
):
    tmp_front_path = None
    tmp_back_path = None
    tmp_passport_path = None

    try:
        if doc_type == "National ID":
            if not id_front or not id_back:
                raise HTTPException(status_code=400, detail="Front and back images required for National ID.")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_front:
                img_front = Image.open(id_front.file)
                img_front.convert('RGB').save(tmp_front, format='JPEG')
                tmp_front_path = tmp_front.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_back:
                img_back = Image.open(id_back.file)
                img_back.convert('RGB').save(tmp_back, format='JPEG')
                tmp_back_path = tmp_back.name

            extractor = EgyptianIDExtractor()
            extracted_data = extractor.extract(front_image=tmp_front_path, back_image=tmp_back_path)
            data_dict = json.loads(extracted_data.to_json())

        elif doc_type == "Passport":
            if not passport:
                raise HTTPException(status_code=400, detail="Passport image required.")
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_pass:
                img_pass = Image.open(passport.file)
                img_pass.convert('RGB').save(tmp_pass, format='JPEG')
                tmp_passport_path = tmp_pass.name

            extractor = PassportExtractor()
            extracted_data = extractor.extract(image=tmp_passport_path)
            data_dict = json.loads(extracted_data.to_json())
        else:
            raise HTTPException(status_code=400, detail="Invalid doc_type. Use 'National ID' or 'Passport'.")

        # Expiry check
        exp_date_str = data_dict.get("expiry_date")
        is_valid = False
        if exp_date_str:
            try:
                exp_date = datetime.strptime(exp_date_str, "%Y/%m/%d").date()
                today = datetime.now().date()
                if exp_date >= today:
                    is_valid = True
            except Exception:
                pass
        
        data_dict["is_valid"] = is_valid
                
        return JSONResponse(content=data_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in [tmp_front_path, tmp_back_path, tmp_passport_path]:
            if path and os.path.exists(path):
                os.unlink(path)

# You can run this with: uvicorn main:app --host 0.0.0.0 --port 8001
