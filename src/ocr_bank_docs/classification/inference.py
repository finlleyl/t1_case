from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch, open_clip
from PIL import Image
from pathlib import Path


device = "cuda" if torch.cuda.is_available() else "cpu"


prompts = ["handwritten text", "printed text"]


def load_clip_model():
    global model, processor
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = AutoModelForZeroShotImageClassification.from_pretrained(
        "openai/clip-vit-large-patch14"
    )


def load_laion_model():
    global model, preprocess_train, preprocess_val, tokenizer, text_features
    model_name = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )  # L2-норма


def classify(cropped_img, classification_model):
    if classification_model == "openai":
        if "processor" not in globals():
            load_clip_model()
        return clip(cropped_img)
    elif classification_model == "hf-hub":
        if "text_features" not in globals():
            load_laion_model()
        return laion(cropped_img)


def clip(cropped_img):
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


def laion(cropped_img):
    # Convert numpy array to PIL Image
    img = Image.fromarray(cropped_img).convert("RGB")
    img_tensor = preprocess_val(img).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        logits = 100.0 * img_feat @ text_features.T
        probs = logits.softmax(dim=-1).squeeze(0)

    label = "handwritten" if probs.argmax().item() == 0 else "printed"
    print(label)
    return label, probs.tolist()
