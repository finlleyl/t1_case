from ultralytics import YOLO
import cv2

def load_model(weights_path: str):
    return YOLO(weights_path)

def detect_from_yolo(model, image, conf=0.5):
    # image — numpy array BGR
    results = model.predict(source=image, conf=conf)[0]
    boxes = []
    for *xyxy, score, cls in results.boxes.data.tolist():
        x1,y1,x2,y2 = xyxy
        boxes.append({
            "x": x1, "y": y1,
            "width": x2-x1, "height": y2-y1,
            "conf": score,
            "class": model.names[int(cls)]
        })
    return boxes

# Пример быстрой проверки
if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    model = YOLO('src/ocr_bank_docs/detection/yolov8_ru_handtext.pt')
    print(detect_from_yolo(model, img))
