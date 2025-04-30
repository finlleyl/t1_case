import cv2
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from src.ocr_bank_docs.classification.inference import classify
from src.ocr_bank_docs.pipelines.run_pipeline import run_pipeline_from_images

router = APIRouter(prefix="/ocr", tags=["optical character recognition"])


@router.post("")
async def ocr(file: UploadFile = File(...)):
    image = await file.read()

    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = await run_pipeline_from_images(img)

    return JSONResponse(content=results)
