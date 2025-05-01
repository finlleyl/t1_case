import json
import argparse
import os
import re
from difflib import SequenceMatcher


def cer(hyp: str, ref: str) -> float:
    """Более корректный CER с Levenshtein расстоянием"""
    m = len(ref) + 1
    n = len(hyp) + 1
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1] / max(1, len(ref))


def wer(hyp: str, ref: str) -> float:
    """Корректный WER с использованием Levenshtein расстояния по словам"""
    hyp_words = hyp.strip().split()
    ref_words = ref.strip().split()

    m = len(ref_words) + 1
    n = len(hyp_words) + 1
    dp = [[0] * n for _ in range(m)]

    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1] / max(1, len(ref_words))


def clean_token(text, mode='digits'):
    if mode == 'digits':
        return ''.join(re.findall(r'\d+', text))
    elif mode == 'letters':
        return ' '.join([w.lower() for w in re.findall(r'\b[а-яА-Яa-zA-Z]{4,}\b', text)])
    return text


def has_fuzzy_digit_match(gt_token, ocr_tokens):
    for ocr_token in ocr_tokens:
        if SequenceMatcher(None, gt_token, ocr_token).ratio() >= 0.6:
            return True
    return False


def filter_gt_combined(gt_data, ocr_data):
    ocr_digit_tokens = set()
    ocr_letter_tokens = set()

    for item in ocr_data:
        ocr_digit_tokens.add(clean_token(item.get('text', ''), 'digits'))
        ocr_letter_tokens.update(clean_token(item.get('text', ''), 'letters').split())

    filtered_gt = []
    removed_texts = []
    for gt_item in gt_data:
        text = gt_item.get('text', '').strip()
        digit_token = clean_token(text, 'digits')
        letter_tokens = clean_token(text, 'letters').split()

        if not digit_token and not letter_tokens:
            removed_texts.append(text)
            continue

        has_digit_match = digit_token and (digit_token in ocr_digit_tokens or has_fuzzy_digit_match(digit_token, ocr_digit_tokens))
        has_letter_match = any(token in ocr_letter_tokens for token in letter_tokens)

        if has_digit_match or has_letter_match:
            filtered_gt.append(gt_item)
        else:
            removed_texts.append(text)

    return filtered_gt, removed_texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR result against ground truth")
    parser.add_argument('--ocr', required=True, help='Path to OCR result JSON (e.g., result.json)')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON (e.g., a_1_1.json)')
    args = parser.parse_args()

    with open(args.ocr, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_data_full = json.load(f)

    ocr_texts = [item['text'] for item in ocr_data if 'text' in item]
    gt_texts_full = [item['text'] for item in gt_data_full if 'text' in item]

    print("\nFULL GT TEXT BEFORE FILTERING:")
    print(' '.join(gt_texts_full))

    print("\nFULL OCR TEXT:")
    print(' '.join(ocr_texts))

    original_len = len(gt_data_full)
    gt_data, removed_texts = filter_gt_combined(gt_data_full, ocr_data)

    print(f"\nFiltered GT blocks: {original_len - len(gt_data)} removed, {len(gt_data)} remaining.")

    if removed_texts:
        print("\nREMOVED TEXTS:")
        for text in removed_texts:
            print(f"- {text}")

    gt_texts = [item['text'] for item in gt_data if 'text' in item]

    print("\nGT TEXT AFTER FILTERING:")
    print(' '.join(gt_texts))

    wer_score = wer(' '.join(ocr_texts), ' '.join(gt_texts))
    cer_score = cer(''.join(ocr_texts), ''.join(gt_texts))

    print(f"\nWER: {wer_score:.4f}")
    print(f"CER: {cer_score:.4f}")


if __name__ == "__main__":
    main()
