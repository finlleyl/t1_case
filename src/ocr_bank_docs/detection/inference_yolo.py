import numpy as np
import torch

_orig_load = torch.load


def _unsafe_load(*args, **kwargs):
    return _orig_load(*args, **{**kwargs, "weights_only": False})


torch.load = _unsafe_load

from ultralytics import YOLO


def load_model(weights_path: str):
    """Загружает модель YOLO по указанному пути."""
    return YOLO(weights_path)


def detect_from_yolo(model, image, conf=0.4):
    """
    Выполняет предсказание модели и возвращает список словарей:
    {"x": float, "y": float, "width": float, "height": float, "conf": float, "class": str}
    """
    results = model.predict(source=image, conf=conf)[0]
    boxes = []
    for *xyxy, score, cls in results.boxes.data.tolist():
        x1, y1, x2, y2 = xyxy
        boxes.append(
            {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "conf": score,
                "class": model.names[int(cls)],
            }
        )
    return sort_boxes(boxes)


def sort_boxes(boxes):
    """
    Сортирует список ограничивающих рамок слева направо по строкам, затем сверху вниз.

    :param boxes: Список словарей с ключами 'x', 'y', 'width', 'height'.
    :return: Отсортированный список словарей.
    """
    # Вычисление центров рамок
    centers = [
        (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2) for box in boxes
    ]

    # Сортировка рамок по вертикальной позиции (y)
    sorted_by_y = sorted(zip(centers, boxes), key=lambda x: x[0][1])

    # Группировка рамок в строки
    lines = []
    current_line = []
    current_y = sorted_by_y[0][0][1]

    for center, box in sorted_by_y:
        if abs(center[1] - current_y) < 10:  # Порог для определения строки
            current_line.append(box)
        else:
            lines.append(
                sorted(current_line, key=lambda x: x["x"])
            )  # Сортировка по x в строке
            current_line = [box]
            current_y = center[1]
    lines.append(
        sorted(current_line, key=lambda x: x["x"])
    )  # Добавление последней строки

    # Сортировка строк по вертикальной позиции
    sorted_boxes = [box for line in lines for box in line]

    return sorted_boxes
