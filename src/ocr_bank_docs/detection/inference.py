import os

import sys

print(sys.executable)


from src.ocr_bank_docs.classification.inference import classify

from .inference_yolo import load_model, detect_from_yolo

weights_path = os.path.join(os.path.dirname(__file__), "yolov8_ru_handtext.pt")
model = load_model(weights_path)


def detect_text_blocks(image):
    return detect_from_yolo(model, image)


def is_handwritten_text(image):
    return True if classify(image)[0] == "handwritten" else False
