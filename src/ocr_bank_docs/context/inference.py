from src.ocr_bank_docs.context.trocr_inference import recognize_text_trocr


def recognize_text(cropped_image):
    return recognize_text_trocr(cropped_image)
