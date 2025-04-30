import torch
_orig_load = torch.load

def _unsafe_load(*args, **kwargs):
    return _orig_load(*args, **{**kwargs, "weights_only": False})

torch.load = _unsafe_load

from ultralytics import YOLO

def load_model(weights_path: str):
    return YOLO(weights_path)



def detect_from_yolo(model, image, conf=0.5):
    results = model.predict(source=image, conf=conf)[0]
    boxes = []
    for *xyxy, score, cls in results.boxes.data.tolist():
        x1, y1, x2, y2 = xyxy
        boxes.append({
            "x": x1, "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "conf": score,
            "class": model.names[int(cls)]
        })
    return boxes

if __name__ == "__main__":
    import cv2, sys
    img = cv2.imread(sys.argv[1])
    model = load_model()
    print(detect_from_yolo(model, img))
