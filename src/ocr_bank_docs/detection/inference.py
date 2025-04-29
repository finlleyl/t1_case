from .inference_yolo import load_model, detect_from_yolo

model = load_model("yolov8_ru_handtext.pt")

def detect_text_blocks(image):
    return detect_from_yolo(model, image)
