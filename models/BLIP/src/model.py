from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipForImageTextRetrieval,
)
import torch


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_blip_captioning(device: str | torch.device | None = None):
    """
    BLIP для генерации подписей к изображениям.
    """
    device = get_device(device)
    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor, device


def load_blip_vqa(device: str | torch.device | None = None):
    """
    BLIP для визуального вопрос-ответа (VQA).
    """
    device = get_device(device)
    model_name = "Salesforce/blip-vqa-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor, device


def load_blip_retrieval(device: str | torch.device | None = None):
    """
    BLIP для image-text retrieval / matching (ITM).
    Возвращает score: насколько текст подходит к картинке.
    """
    device = get_device(device)
    model_name = "Salesforce/blip-itm-base-coco"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor, device
