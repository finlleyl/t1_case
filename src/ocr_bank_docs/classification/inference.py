import torch
import open_clip
from PIL import Image
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name, device=device
)
tokenizer = open_clip.get_tokenizer(model_name)

prompts = ["handwritten text", "printed text"]

with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
    text_tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


def classify(cropped_img):
    # Convert numpy array to PIL Image
    img = Image.fromarray(cropped_img).convert("RGB")
    img_tensor = preprocess_val(img).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        logits = 100.0 * img_feat @ text_features.T
        probs = logits.softmax(dim=-1).squeeze(0)

    label = "handwritten" if probs.argmax().item() == 0 else "printed"
    return label, probs.tolist()
