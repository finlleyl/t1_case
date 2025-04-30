import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import json
import io

API_URL = "http://localhost:8000/ocr"

def draw_bboxes_on_image(image, json_data):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for item in json_data:
        bbox = item["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        draw.text((x, y - 10), item["text"], font=font, fill="red")
    return image

def main():
    st.title("OCR с FastAPI и Streamlit")

    # Инициализация session_state
    if 'json_data' not in st.session_state:
        st.session_state.json_data = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None

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
                    st.session_state.json_data = response.json()
                    st.session_state.annotated_image = draw_bboxes_on_image(image.copy(), st.session_state.json_data)

        # Показываем результаты, если они есть в session_state
        if st.session_state.annotated_image is not None:
            st.image(
                st.session_state.annotated_image,
                caption="Изображение с bounding box'ами",
                use_column_width=True,
            )
        
        if st.session_state.json_data is not None:
            st.download_button(
                label="Скачать JSON",
                data=json.dumps(st.session_state.json_data, ensure_ascii=False),
                file_name="ocr_result.json",
                mime="application/json",
            )
            st.json(st.session_state.json_data)

if __name__ == "__main__":
    main()
