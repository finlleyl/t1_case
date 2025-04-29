def wer(hyp: str, ref: str) -> float:
    # Разбиваем строки на слова
    hyp_words = hyp.split()
    ref_words = ref.split()

    dp = [[0] * (len(ref_words) + 1) for _ in range(len(hyp_words) + 1)]

    for i in range(len(hyp_words) + 1):
        dp[i][0] = i
    for j in range(len(ref_words) + 1):
        dp[0][j] = j

    for i in range(1, len(hyp_words) + 1):
        for j in range(1, len(ref_words) + 1):
            if hyp_words[i - 1] == ref_words[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Вставка
                dp[i][j - 1] + 1,  # Удаление
                dp[i - 1][j - 1] + cost  # Замена
            )

    # Подсчет ошибок
    errors = dp[-1][-1]
    total_words = len(ref_words)

    return (errors / total_words) if total_words != 0 else 0.0


def cer(hyp: str, ref: str) -> float:
    # Преобразуем строки в списки символов
    hyp_chars = list(hyp)
    ref_chars = list(ref)

    dp = [[0] * (len(ref_chars) + 1) for _ in range(len(hyp_chars) + 1)]

    for i in range(len(hyp_chars) + 1):
        dp[i][0] = i
    for j in range(len(ref_chars) + 1):
        dp[0][j] = j

    for i in range(1, len(hyp_chars) + 1):
        for j in range(1, len(ref_chars) + 1):
            if hyp_chars[i - 1] == ref_chars[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Вставка
                dp[i][j - 1] + 1,  # Удаление
                dp[i - 1][j - 1] + cost  # Замена
            )

    # Подсчет ошибок
    errors = dp[-1][-1]
    total_chars = len(ref_chars)

    return (errors / total_chars) if total_chars != 0 else 0.0

#-----------------------------------------------------------------------
# Тестовые данные WER
reference = "the cat sat on the mat"
hypothesis = "the cat sat mat"

#  WER
wer_score = wer(hypothesis, reference)
print(f"WER: {wer_score:.2f}%")  # Ожидаемый результат: 33.33%

# Тестовые данные CER
reference_cer = "hello"
hypothesis_cer = "hallo"

# Расчет CER
cer_score = cer(hypothesis_cer, reference_cer)
print(f"CER: {cer_score:.2f}%")  # Ожидаемый результат: 20.00%