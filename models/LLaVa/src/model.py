from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

def load_llava(device: str | torch.device = "cpu"):

    model_name = "llava-hf/llava-1.5-7b-hf"  # маленькая (7B) версия LLaVA с HF
    model = AutoModelForVision2Seq.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def llava_ask(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 128,
):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # убираем входные токены, оставляем только сгенерированный ответ
    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer.strip()
