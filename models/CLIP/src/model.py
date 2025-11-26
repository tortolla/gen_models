from transformers import CLIPModel, CLIPProcessor
import torch

def load_clip(device: str | torch.device = "cpu"):


    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor
