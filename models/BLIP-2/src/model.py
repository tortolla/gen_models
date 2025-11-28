# src/model.py

from __future__ import annotations

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Можно поменять на другой чекпоинт, например:
# "Salesforce/blip2-flan-t5-xl"
DEFAULT_BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"


def load_blip2(
    model_name: str = DEFAULT_BLIP2_MODEL,
    device: str | torch.device | None = None,
):
    """
    Загружает BLIP-2 с HuggingFace и возвращает (model, processor, device).

    - автоматически выбирает cuda/cpu
    - на cuda использует float16, на cpu — float32
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    model.to(device)
    model.eval()

    return model, processor, device


def generate_answer(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
) -> str:
    """
    Один вызов BLIP-2: (картинка + текстовый промпт) -> текстовый ответ.
    """
    inputs = processor(
        image,
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()
