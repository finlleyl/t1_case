import argparse
import json
import cv2

from ocr_bank_docs.detection.inference import detect_and_recognize

def run_pipeline_from_images(img):
    """
    Получает изображение (BGR), выполняет детекцию YOLO и возвращает список результатов.
    """
    if img is None:
        print("Не удалось загрузить изображение.")
        return

    detections = detect_and_recognize(img)
    if not detections:
        print("Не удалось обнаружить текстовые блоки.")
        return

    return detections


def main():
    print("Запуск OCR Pipeline...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Путь до файла.")
    parser.add_argument("--out", type=str, required=True, help="Путь до выходного файла.")
    args = parser.parse_args()

    img = cv2.imread(args.img)
    result = run_pipeline_from_images(img)
    if result is None:
        print("Не удалось выполнить OCR Pipeline.")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
