# Для запуска python evaluate.py --pred rec.json --gt gt.json
#!/usr/bin/env python3
import argparse
import json
from metriki import wer, cer

def load_json_data(file_path: str) -> dict:
    """Загружает JSON-файл с текстами"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_text(text: str) -> str:
    """Предобработка текста для вычисления метрик"""
    text = text.lower().strip()
    text = ' '.join(text.split())  # Удаляем лишние пробелы
    return text

def main():
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description='Evaluate OCR results')
    parser.add_argument('--pred', type=str, required=True, help='Path to predictions JSON')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth JSON')
    args = parser.parse_args()

    # Загрузка данных
    pred_data = load_json_data(args.pred)
    gt_data = load_json_data(args.gt)

    total_wer, total_cer, count = 0.0, 0.0, 0

    for doc_id in gt_data:
        if doc_id not in pred_data:
            print(f"Warning: No prediction for document {doc_id}")
            continue

        gt_text = preprocess_text(gt_data[doc_id])
        pred_text = preprocess_text(pred_data[doc_id])

        # Вычисление метрик
        try:
            total_wer += wer(pred_text, gt_text)  # Аргументы hyp и ref
            total_cer += cer(pred_text, gt_text)
            count += 1
        except ZeroDivisionError:
            print(f"Warning: Empty ground truth for document {doc_id}")

    # Расчет средних значений
    avg_wer = total_wer / count if count > 0 else 0.0
    avg_cer = total_cer / count if count > 0 else 0.0

    # Вывод результатов
    print(f"WER: {avg_wer:.2f}")
    print(f"CER: {avg_cer:.2f}")

if __name__ == "__main__":
    main()