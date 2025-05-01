from io import BytesIO
import json
import cv2
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from matplotlib import pyplot as plt
import numpy as np
from src.ocr_bank_docs.classification.inference import classify
from src.ocr_bank_docs.pipelines.run_pipeline import run_pipeline_from_images
from PIL import Image, ImageDraw, ImageFont

router = APIRouter(prefix="/ocr", tags=["optical character recognition"])


@router.post("")
async def ocr(classification_model: str = Form(...), file: UploadFile = File(...)):
    image = await file.read()

    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = await run_pipeline_from_images(img, classification_model)

    return JSONResponse(content=results)


@router.post("/bbox/")
async def create_upload_file(file: UploadFile = File(...), json_data: str = ""):
    # Load image from file
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    # Load JSON data
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

    # Create a copy of the image for drawing bounding boxes and text
    img_with_boxes = image.copy()

    # Iterate through each element in the JSON
    for entry in data:
        bbox = entry.get("bbox")
        text = entry.get("text")

        # Extract the bounding box coordinates
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # Draw the bounding box on the image
        cv2.rectangle(
            img_with_boxes, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
        )

        # Place the text near the box using PIL
        img_with_boxes_pil = Image.fromarray(img_with_boxes)
        draw = ImageDraw.Draw(img_with_boxes_pil)
        try:
            # Try system fonts, fallback to default if not available
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
            )
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except OSError:
                font = ImageFont.load_default()

        draw.text((int(x), int(y - 30)), text, font=font, fill=(255, 0, 0))
        img_with_boxes = np.array(img_with_boxes_pil)

    # Convert image to displayable format
    img_with_boxes = Image.fromarray(img_with_boxes)

    # Show the image with bounding boxes
    plt.imshow(img_with_boxes)
    plt.axis("off")
    plt.show()

    return HTMLResponse(
        content="File uploaded and processed successfully", status_code=200
    )
