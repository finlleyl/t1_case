from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Загрузка модели и процессора
processor = TrOCRProcessor.from_pretrained("raxtemur/trocr-base-ru")
model = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru")



def recognize_text_trocr(cropped_image):
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(cropped_image)
    
    # Подготовка изображения
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    # Генерация текста
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(generated_text)
    return generated_text
