# src/model.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image


class StableVideoDiffusionImg2Vid:
    """
    Простая обёртка над StableVideoDiffusionPipeline (img2vid-xt-1-1).

    Использование:
        model = StableVideoDiffusionImg2Vid()
        frames = model.generate("data/input.png", output_path="data/output.mp4")
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
        enable_cpu_offload: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            variant=variant,
        )

        if device == "cuda" and enable_cpu_offload:
            # экономим VRAM (diffusers сам гоняет модули между CPU/GPU)
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

    @torch.inference_mode()
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        num_frames: int = 25,
        num_inference_steps: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        decode_chunk_size: int = 8,
        seed: Optional[int] = 42,
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[Image.Image]:
        """
        image:           путь к картинке или PIL.Image
        num_frames:      сколько кадров генерировать (SVD-XT умеет до 25)
        num_steps:       шаги диффузии
        fps:             частота кадров результирующего видео
        motion_bucket_id: управляет силой движения (чем больше, тем больше движения)
        noise_aug_strength: сколько шума добавлять к кондиционирующему кадру
        *guidance*:      min/max guidance scale по фреймам (линейно)
        decode_chunk_size: сколько кадров декодировать VAE за раз (для экономии памяти)
        seed:            random seed
        output_path:     если задан — сохранить .mp4 через export_to_video
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # рекомендуемый размер 1024x576 для SVD :contentReference[oaicite:0]{index=0}
        image = image.resize((1024, 576))

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            image=image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
        )

        frames: List[Image.Image] = result.frames[0]

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=fps)

        return frames
