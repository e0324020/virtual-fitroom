"""Self-contained CatVTON inference (mask-based).

Reimplements Zheng-Chong/CatVTON's mask-based pipeline using the same base model
(SD1.5-inpainting) and fine-tuned self-attention weights (zhengchong/CatVTON),
without cloning the original repo. Adapted from:
  - model/pipeline.py
  - model/attn_processor.py
  - utils.py

The trick: person + garment are concatenated along the height axis and fed to an
SD1.5 inpainting UNet whose self-attention layers were fine-tuned (cross-attention
is bypassed). Only ~50M params change; the base stays SD1.5.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file

BASE_CKPT = "runwayml/stable-diffusion-inpainting"
VAE_CKPT = "stabilityai/sd-vae-ft-mse"
ATTN_REPO = "zhengchong/CatVTON"


class SkipAttnProcessor(torch.nn.Module):
    """Bypass attention entirely (used for the unused text cross-attention)."""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        return hidden_states


# ---------------------------------------------------------------- image helpers


def resize_and_crop(image, size):
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_w, new_h = w, w * target_h // target_w
    else:
        new_h, new_w = h, h * target_w // target_h
    image = image.crop(((w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2))
    return image.resize(size, Image.LANCZOS)


def resize_and_padding(image, size):
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h:
        new_h, new_w = target_h, w * target_h // h
    else:
        new_w, new_h = target_w, h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def prepare_image(image):
    if isinstance(image, Image.Image):
        image = [image]
    image = [np.array(i.convert("RGB"))[None, :] for i in image]
    image = np.concatenate(image, axis=0)
    image = image.transpose(0, 3, 1, 2)
    return torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0


def prepare_mask_image(mask_image):
    if isinstance(mask_image, Image.Image):
        mask_image = [mask_image]
    mask_image = np.concatenate(
        [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0
    )
    mask_image = mask_image.astype(np.float32) / 255.0
    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1
    return torch.from_numpy(mask_image)


def compute_vae_encodings(image, vae):
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    return model_input * vae.config.scaling_factor


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


# ---------------------------------------------------------------- the pipeline


class CatVTON:
    def __init__(self, attn_version="mix-48k-1024", dtype=torch.bfloat16, device="cuda"):
        self.device = device
        self.dtype = dtype

        self.noise_scheduler = DDIMScheduler.from_pretrained(BASE_CKPT, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(VAE_CKPT, torch_dtype=dtype).to(device)

        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                BASE_CKPT, subfolder="unet", torch_dtype=dtype, variant="fp16"
            )
        except Exception:
            self.unet = UNet2DConditionModel.from_pretrained(
                BASE_CKPT, subfolder="unet", torch_dtype=dtype
            )
        self.unet.to(device)

        self._apply_attention_slicing()
        self._init_adapter()
        self._load_attn_weights(attn_version)

        self.unet.eval()
        self.vae.eval()
        try:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
        except Exception:
            pass

    def _apply_attention_slicing(self):
        # NOTE: must run *before* _init_adapter, because slicing replaces the
        # attention processors (which would otherwise clobber our Skip processors).
        for fn, args in (
            ("enable_attention_slicing", ("max",)),
            ("set_attention_slice", ("max",)),
            ("set_attention_slicing", ("max",)),
        ):
            if hasattr(self.unet, fn):
                try:
                    getattr(self.unet, fn)(*args)
                    return
                except Exception:
                    continue

    def _init_adapter(self):
        procs = dict(self.unet.attn_processors)
        new = {}
        for name, proc in procs.items():
            new[name] = SkipAttnProcessor() if name.endswith("attn2.processor") else proc
        self.unet.set_attn_processor(new)

    def _load_attn_weights(self, version):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(ATTN_REPO, f"{version}/attention/model.safetensors")
        sd = load_file(path)

        attn1 = [m for n, m in self.unet.named_modules() if n.split(".")[-1] == "attn1"]
        prefixes = sorted({k.split(".")[0] for k in sd}, key=int)
        if len(prefixes) != len(attn1):
            raise RuntimeError(
                f"CatVTON attn mismatch: {len(prefixes)} checkpoint groups vs "
                f"{len(attn1)} self-attn modules"
            )

        loaded = 0
        for prefix, module in zip(prefixes, attn1):
            for k, v in sd.items():
                if not k.startswith(prefix + "."):
                    continue
                obj = module
                parts = k[len(prefix) + 1 :].split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                param = getattr(obj, parts[-1])
                param.data.copy_(v.to(param.dtype))
                loaded += 1
        print(f"[CatVTON] loaded {loaded}/{len(sd)} attention tensors across {len(attn1)} modules")

    @torch.no_grad()
    def __call__(
        self,
        person: Image.Image,
        cloth: Image.Image,
        mask: Image.Image,
        num_inference_steps: int = 30,
        guidance_scale: float = 2.5,
        height: int = 768,
        width: int = 576,
        generator=None,
    ):
        concat_dim = -2
        person = resize_and_crop(person.convert("RGB"), (width, height))
        mask = resize_and_crop(mask.convert("L"), (width, height))
        cloth = resize_and_padding(cloth.convert("RGB"), (width, height))

        image = prepare_image(person).to(self.device, dtype=self.dtype)
        condition_image = prepare_image(cloth).to(self.device, dtype=self.dtype)
        mask_t = prepare_mask_image(mask).to(self.device, dtype=self.dtype)

        masked_image = image * (mask_t < 0.5)

        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = F.interpolate(mask_t, size=masked_latent.shape[-2:], mode="nearest")
        del image, cloth, condition_image

        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        latents = torch.randn(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.dtype,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(condition_latent)], dim=concat_dim),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)
        del masked_latent, condition_latent, mask_latent, mask_t, masked_image

        for i, t in enumerate(timesteps):
            model_input = torch.cat([latents] * 2) if do_cfg else latents
            model_input = self.noise_scheduler.scale_model_input(model_input, t)
            inpaint_input = torch.cat(
                [model_input, mask_latent_concat, masked_latent_concat], dim=1
            )
            noise_pred = self.unet(
                inpaint_input, t.to(self.device), encoder_hidden_states=None, return_dict=False
            )[0]
            if do_cfg:
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (text - uncond)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return numpy_to_pil(image)[0]


_pipeline: Optional[CatVTON] = None


def get_pipeline() -> CatVTON:
    global _pipeline
    if _pipeline is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _pipeline = CatVTON(dtype=dtype)
    return _pipeline
