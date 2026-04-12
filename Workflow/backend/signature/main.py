import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ── Custom Layer ─────────────────────────────────────────────────────────────
class AbsoluteDifferenceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# ── Constants & Config ────────────────────────────────────────────────────────
IMG_SIZE = 128
MODEL_REPO = "Mels22/Signature-Detection-Verification"
DETECTOR_FILENAME = "detector_yolo_1cls.pt"
KERAS_MODEL_FILENAME = "siamese_signature_model.keras"

app = FastAPI(title="Signature Verification Service")

# Global models
yolo_model = None
keras_model = None

@app.on_event("startup")
async def load_models():
    global yolo_model, keras_model
    
    # Load YOLO
    if not os.path.exists(DETECTOR_FILENAME):
        print(f"Downloading {DETECTOR_FILENAME} from Hub ...")
        hf_hub_download(repo_id=MODEL_REPO, filename=DETECTOR_FILENAME, local_dir=".")
    yolo_model = YOLO(DETECTOR_FILENAME)
    
    # Load Keras
    if not os.path.exists(KERAS_MODEL_FILENAME):
        print(f"Downloading {KERAS_MODEL_FILENAME} from Hub ...")
        hf_hub_download(repo_id=MODEL_REPO, filename=KERAS_MODEL_FILENAME, local_dir=".")
    
    keras_model = tf.keras.models.load_model(
        KERAS_MODEL_FILENAME,
        safe_mode=False,
        custom_objects={"AbsoluteDifferenceLayer": AbsoluteDifferenceLayer},
    )

def preprocess_for_keras(pil_img, target=(IMG_SIZE, IMG_SIZE)):
    img = np.array(pil_img.convert("L"))
    img = cv2.resize(img, target).astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def crop_from_xywhn(image_pil, xywhn_box, padding=10):
    W, H = image_pil.size
    xc, yc, w, h = xywhn_box
    x1 = max(int((xc - w / 2) * W) - padding, 0)
    y1 = max(int((yc - h / 2) * H) - padding, 0)
    x2 = min(int((xc + w / 2) * W) + padding, W)
    y2 = min(int((yc + h / 2) * H) + padding, H)
    return image_pil.crop((x1, y1, x2, y2))

@app.post("/verify")
async def verify_signature(
    document: UploadFile = File(...),
    reference: UploadFile = None,
    threshold: float = Form(0.85),
    det_conf: float = Form(0.4)
):
    try:
        # Load reference image
        if reference:
            ref_bytes = await reference.read()
            ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        else:
            # Look in reference folder
            ref_dir = "reference"
            if not os.path.exists(ref_dir):
                os.makedirs(ref_dir)
            
            ref_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not ref_files:
                raise HTTPException(status_code=400, detail="No reference signature found in reference folder.")
            
            ref_path = os.path.join(ref_dir, ref_files[0])
            ref_img = Image.open(ref_path).convert("RGB")

        doc_bytes = await document.read()
        doc_img = Image.open(io.BytesIO(doc_bytes)).convert("RGB")
        
        # 1. Detection
        results = yolo_model.predict(doc_img, conf=det_conf, verbose=False)[0]
        
        detections = []
        img_ref_prep = preprocess_for_keras(ref_img)
        
        for box in results.boxes:
            bbox = box.xywhn[0].tolist()
            crop = crop_from_xywhn(doc_img, bbox)
            
            # 2. Verification
            img_test_prep = preprocess_for_keras(crop)
            
            score = float(keras_model.predict(
                [np.expand_dims(img_ref_prep, 0), np.expand_dims(img_test_prep, 0)],
                verbose=0
            )[0][0])
            
            # Convert crop to base64 for display in frontend
            buffered = io.BytesIO()
            crop.save(buffered, format="PNG")
            crop_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            detections.append({
                'bbox': bbox,
                'similarity': score,
                'is_genuine': score >= threshold,
                'crop_base64': crop_base64
            })
            
        return {
            "detections": detections,
            "threshold": threshold,
            "status": "success",
            "reference_used": "local_folder" if not reference else "uploaded"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
