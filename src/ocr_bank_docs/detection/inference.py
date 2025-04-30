import os

from .inference_yolo import load_model, detect_from_yolo

weights_path = os.path.join(os.path.dirname(__file__), "yolov8_ru_handtext.pt")
model = load_model(weights_path)

def detect_text_blocks(image):
    return detect_from_yolo(model, image)
