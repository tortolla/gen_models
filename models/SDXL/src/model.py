# src/model.py
import torch
from diffusers import StableDiffusionXLPipeline


class SDXLModel:
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        self.pipe.to(self.device)

        # немного ускорения
        self.pipe.enable_attention_slicing()
        if self.device == "cuda":
            self.pipe.enable_xformers_memory_efficient_attention = getattr(
                self.pipe, "enable_xformers_memory_efficient_attention", lambda: None
            )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        seed: int | None = 42,
    ):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        # diffusers для SDXL возвращает .images = список PIL.Image
        return output.images
