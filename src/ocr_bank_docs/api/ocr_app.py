import streamlit as st
import requests
from PIL import Image, ImageDraw
import json
import io

API_URL = "http://localhost:8000/ocr"


def draw_bboxes_on_image(image, json_data):
    draw = ImageDraw.Draw(image)
    for item in json_data:
        bbox = item["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw.text((x, y - 10), item["text"], fill="red")
    return image


def main():
    st.title("OCR с FastAPI и Streamlit")

    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)

        if st.button("Извлечь текст"):
            with st.spinner("Обработка..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    json_data = response.json()
                    annotated_image = draw_bboxes_on_image(image.copy(), json_data)
                    st.image(
                        annotated_image,
                        caption="Изображение с bounding box'ами",
                        use_column_width=True,
                    )
                    st.download_button(
                        label="Скачать JSON",
                        data=json.dumps(json_data, ensure_ascii=False),
                        file_name="ocr_result.json",
                        mime="application/json",
                    )
                    st.json(json_data)
                else:
                    st.error(f"Ошибка {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
