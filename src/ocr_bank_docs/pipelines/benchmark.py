import time
import glob
import os
import asyncio
from pathlib import Path
import cv2

from src.ocr_bank_docs.pipelines.run_pipeline import run_pipeline_from_images


async def benchmark_pipeline(image_dir: str, num_images: int = 10):
    base_dir = Path(__file__).parent.resolve()
    dir_path = (base_dir / image_dir).resolve()
    print(f"Ищем в {dir_path}")
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    images = []
    for ext in exts:
        images.extend(dir_path.glob(ext))
    images = images[:num_images]
    if not images:
        print("Нет картинок — проверьте путь!")
        return

    total_start = time.perf_counter()
    for img_path in images:
        img = cv2.imread(str(img_path))
        start = time.perf_counter()
        result = await run_pipeline_from_images(img, "openai")
        elapsed = time.perf_counter() - start
        print(f"{img_path.name}: {elapsed:.3f}s")
    total = time.perf_counter() - total_start
    print(f"Всего: {total:.3f}s, в среднем: {total/len(images):.3f}s")


if __name__ == "__main__":
    asyncio.run(benchmark_pipeline("handwritten_test_images", num_images=10))
