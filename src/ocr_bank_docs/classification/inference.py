import torch
import open_clip
from PIL import Image
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "openai/clip-vit-large-patch14"
# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#     model_name, device=device
# )
# tokenizer = open_clip.get_tokenizer(model_name)

# Load model directly
from transformers import CLIPProcessor, CLIPModel
import requests


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

prompts = ["handwritten text", "printed text"]

def classify(cropped_img):
    # Convert numpy array to PIL Image
    img = Image.fromarray(cropped_img).convert("RGB")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        inputs = processor(
            text=prompts,
            images=img,
            return_tensors="pt",
            padding=True,
        )

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze(0)

    label = "handwritten" if probs.argmax().item() == 0 else "printed"
    print(label)
    return label, probs.tolist()

# text_tokens = tokenizer(prompts).to(device)
# text_features = model.encode_text(text_tokens)
# text_features = text_features / text_features.norm(dim=-1, keepdim=True)


# def classify(cropped_img):
#     # Convert numpy array to PIL Image
#     img = Image.fromarray(cropped_img).convert("RGB")
#     img_tensor = preprocess_val(img).unsqueeze(0).to(device)

#     with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
#         img_feat = model.encode_image(img_tensor)
#         img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

#         logits = 100.0 * img_feat @ text_features.T
#         probs = logits.softmax(dim=-1).squeeze(0)

#     label = "handwritten" if probs.argmax().item() == 0 else "printed"
#     print(label)
#     return label, probs.tolist()
