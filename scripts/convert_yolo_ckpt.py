# scripts/export_ultra_ckpt.py
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / \
      "src/ocr_bank_docs/detection/yolov8_ru_handtext.pt"
DST = SRC.with_name("yolov8_ru_handtext_ultra.pt")

add_safe_globals([DetectionModel])
full = torch.load(SRC, map_location="cpu", weights_only=False)

# гарантируем, что full["model"] — это сам DetectionModel
model_obj = full.get("model", full)  # если там уже DetectionModel
new_ckpt = {"model": model_obj}
torch.save(new_ckpt, DST)
print("Saved full Ultralytics checkpoint to", DST)
