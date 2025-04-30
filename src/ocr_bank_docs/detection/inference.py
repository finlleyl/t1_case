import numpy as np
from .inference_yolo import load_model, detect_from_yolo

# Конфигурация YOLO
YOLO_WEIGHTS = "src/ocr_bank_docs/detection/yolov8_ru_handtext.pt"
YOLO_CONF_THRESHOLD = 0.4

# Инициализация модели
yolo_model = load_model(YOLO_WEIGHTS)

def detect_and_recognize(image: np.ndarray):
    """
    Выполняет только детекцию боксов через YOLO и возвращает список словарей:
    {"bbox": {x, y, width, height}, "conf": float, "class": str}
    """
    yolo_boxes = detect_from_yolo(yolo_model, image, conf=YOLO_CONF_THRESHOLD)
    results = []
    for yb in yolo_boxes:
        results.append({
            "bbox": {
                "x": float(yb["x"]),
                "y": float(yb["y"]),
                "width": float(yb["width"]),
                "height": float(yb["height"])
            },
            "conf": float(yb["conf"]),
            "class": yb.get("class")
        })
    return results
