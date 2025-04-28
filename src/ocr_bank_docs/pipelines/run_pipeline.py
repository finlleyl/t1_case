import argparse
import json

import cv2


from ..detection.inference import detect_text_blocks
from ..context.inference import recognize_text


def main():
    print("Запуск OCR Pipeline...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Путь до файла.")
    parser.add_argument(
        "--out", type=str, required=True, help="Путь до выходного файла."
    )
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        print("Не удалось загрузить изображение.")
        return

    if img is None:
        print("Не удалось загрузить изображение.")
        return

    boxes = detect_text_blocks(img)
    if boxes is None:
        print("Не удалось обнаружить текстовые блоки.")
        return

    result = []

    for bbox in boxes:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cropped = img[int(y) : int(y + h), int(x) : int(x + w)]

        text = recognize_text(cropped)
        if text is None:
            print("Не удалось распознать текст.")
            return

        result.append({"bbox": {"x": x, "y": y, "width": w, "height": h}, "text": text})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
