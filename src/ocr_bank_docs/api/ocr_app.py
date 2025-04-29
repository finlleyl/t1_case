import asyncio

import streamlit as st
import cv2
import numpy as np
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.ocr_bank_docs.pipelines.run_pipeline import run_pipeline_from_images

def show_image_with_bboxes(image, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    return image

async def main():
    st.title("OCR")
    uploaded_file = st.file_uploader("Загрузите изображение", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        results = await run_pipeline_from_images(img)

        img_with_boxes = show_image_with_bboxes(img.copy(), [res["bbox"] for res in results])

        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Распознанное изображение с блоками текста", use_container_width=True)

        st.subheader("Результат OCR в формате JSON")
        st.json(results)

        if st.button("Скачать результат"):
            result_json = json.dumps(results, ensure_ascii=False, indent=4)
            st.download_button("Скачать Json", result_json, file_name="ocr_result.json", mime="application/json")


if __name__ == "__main__":
    asyncio.run(main())