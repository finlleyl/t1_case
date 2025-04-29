import os
import shutil
import random
import pathlib
from pathlib import Path

def prepare_yolo_dataset(src_root: str, dst_root: str, val_ratio: float = 0.2):
    src_images = Path(src_root) / "images"
    src_labels = Path(src_root) / "labels"
    dst_root = Path(dst_root)

    # Создание директорий
    for split in ["train", "val"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Собираем список изображений
    img_paths = list(src_images.glob("*.[jp][pn]g"))  # .jpg, .png
    random.shuffle(img_paths)
    val_count = int(len(img_paths) * val_ratio)
    val_imgs = set(img_paths[:val_count])

    for img_path in img_paths:
        split = "val" if img_path in val_imgs else "train"
        label_path = src_labels / (img_path.stem + ".txt")

        # Куда копировать
        dst_img = dst_root / "images" / split / img_path.name
        dst_lbl = dst_root / "labels" / split / label_path.name

        shutil.copy2(img_path, dst_img)
        if label_path.exists():
            shutil.copy2(label_path, dst_lbl)

    # Создаём data.yaml
    data_yaml = dst_root / "data.yaml"
    data_yaml.write_text(
        f"path: {dst_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['text']\n"
    )

    print(f"✅ Датасет подготовлен в {dst_root}. Train: {len(img_paths) - val_count}, Val: {val_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Путь к папке с images/ и labels/")
    parser.add_argument("--dst", required=True, help="Куда сохранить подготовленный датасет")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Доля валидации (по умолчанию 0.2)")
    args = parser.parse_args()

    prepare_yolo_dataset(args.src, args.dst, args.val_ratio)
