import cv2
import numpy as np
import streamlit as st

import sys, os
# добавляем папку src/ в начало поиска модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from ocr_bank_docs.detection.inference import detect_and_recognize

# Настройка страницы
st.set_page_config(page_title="YOLO Detection", layout="wide")
st.title("YOLO Detection — T1 Case")

# Функция отрисовки боксов YOLO
def draw_boxes(img, boxes):
    vis = img.copy()
    for bb in boxes:
        x = int(bb["bbox"]["x"])
        y = int(bb["bbox"]["y"])
        w = int(bb["bbox"]["width"])
        h = int(bb["bbox"]["height"])
        # рисуем bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # подпись класс + conf
        label = f"{bb.get('class','obj')} {bb['conf']:.2f}"
        cv2.putText(vis, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # конвертируем BGR->RGB для корректного отображения
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# Загрузка изображения через Streamlit
uploaded = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"]);
if not uploaded:
    st.info("Ожидание загрузки изображения...")
    st.stop()

# Декодируем в OpenCV-формат (BGR)
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Детекция через YOLO
results = detect_and_recognize(img)

# Визуализация
st.subheader("Результат детекции YOLO")
vis_img = draw_boxes(img, results)
st.image(vis_img, use_column_width=True)

# Вывод JSON с результатами
st.subheader("Данные детекции (JSON)")
st.json(results)
