#!/usr/bin/env python3
"""Generate poster images with Flex.2 + Canny control and optional trained LoRA."""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "YES")
sys.path.insert(0, os.getcwd())

import torch
from safetensors.torch import load_file

from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.util.get_model import get_model_class

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LORA_FILE = "lora_model.safetensors"
DEFAULT_HF_LORA = "libadi/flex2.poster/lora_model.safetensors"
LOCAL_LORA_CANDIDATES = (
    REPO_ROOT / "ckpts" / DEFAULT_LORA_FILE,
    REPO_ROOT / "ckpts" / "flex2_poster_lora_bilingual.safetensors",
    REPO_ROOT / "output/lora/flex2_poster_lora_bilingual/flex2_poster_lora_bilingual_000002500.safetensors",
)


def download_lora_from_hub(spec: str) -> Path:
    from huggingface_hub import hf_hub_download

    parts = spec.split("/")
    if len(parts) == 2:
        repo_id, filename = spec, DEFAULT_LORA_FILE
    elif len(parts) >= 3:
        repo_id = "/".join(parts[:2])
        filename = parts[-1]
    else:
        raise ValueError(
            f"Invalid Hugging Face LoRA spec: {spec!r}. "
            "Use 'user/repo' or 'user/repo/filename.safetensors'."
        )

    print(f"Downloading LoRA from Hugging Face: {repo_id}/{filename}")
    return Path(hf_hub_download(repo_id, filename=filename))


def resolve_lora_path(lora_arg):
    if lora_arg in (None, False):
        return None

    if lora_arg == "default":
        for path in LOCAL_LORA_CANDIDATES:
            if path.is_file():
                return path
        if DEFAULT_HF_LORA:
            return download_lora_from_hub(DEFAULT_HF_LORA)
        raise FileNotFoundError(
            "No local LoRA found. Pass --lora libadi/flex2.poster/lora_model.safetensors"
        )

    path = Path(lora_arg).expanduser()
    if path.is_file():
        return path

    if lora_arg.startswith("hf:"):
        return download_lora_from_hub(lora_arg.removeprefix("hf:"))

    if "/" in lora_arg and not lora_arg.startswith((".", "/")):
        return download_lora_from_hub(lora_arg)

    raise FileNotFoundError(f"LoRA not found: {lora_arg}")

def attach_lora(sd, lora_path):
    cfg = NetworkConfig(type="lora", linear=16, linear_alpha=16, transformer_only=True)
    kwargs = {}
    if hasattr(sd, "target_lora_modules"):
        kwargs["target_lin_modules"] = sd.target_lora_modules

    network = LoRASpecialNetwork(
        text_encoder=sd.text_encoder,
        unet=sd.get_model_to_train(),
        lora_dim=cfg.linear,
        multiplier=1.0,
        alpha=cfg.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        is_flux=sd.model_config.is_flux,
        network_config=cfg,
        network_type=cfg.type,
        transformer_only=cfg.transformer_only,
        is_transformer=sd.is_transformer,
        base_model=sd,
        **kwargs,
    )
    network.force_to(sd.device_torch, dtype=torch.float32)
    network.apply_to(sd.text_encoder, sd.get_model_to_train(), False, True)
    network._update_torch_multiplier()
    network.load_weights(load_file(str(lora_path)))
    network.eval()
    network.is_active = True
    sd.network = network


def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = ModelConfig(
        name_or_path="ostris/Flex.2-preview",
        arch="flex2",
        quantize=True,
        quantize_te=True,
        dtype="bf16",
        model_kwargs={
            "control_dropout": 0.0,
            "inpaint_dropout": 1.0,
            "do_random_inpainting": False,
            "inpaint_random_chance": 0.0,
            "invert_inpaint_mask_chance": 0.0,
            "random_blur_mask": False,
            "random_dialate_mask": False,
        },
    )
    model_class = get_model_class(model_config)
    sd = model_class(
        device=device,
        model_config=model_config,
        dtype="bf16",
        noise_scheduler=model_class.get_train_scheduler(),
    )
    sd.load_model()
    return sd


def build_prompt(prompt: str, canny_path: Path | None = None) -> str:
    prompt = (prompt or "").strip()
    if canny_path is None:
        return prompt
    return f"{prompt} --ctrl_idx 1 --ctrl_img {canny_path.resolve()}".strip()


def generate_poster(
    sd,
    prompt: str,
    canny_path: Path | None,
    output_path: Path,
    width: int = 768,
    height: int = 768,
    seed: int = 42,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 25,
    network_multiplier: float = 1.0,
):
    if canny_path:
        canny_path = Path(canny_path).expanduser()
        if not canny_path.is_file():
            raise FileNotFoundError(f"Canny image not found: {canny_path}")
    else:
        canny_path = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_kwargs = {}
    if canny_path is not None:
        config_kwargs["ctrl_idx"] = 1

    with torch.no_grad():
        sd.generate_images([
            GenerateImageConfig(
                prompt=build_prompt(prompt, canny_path),
                width=width,
                height=height,
                negative_prompt="",
                seed=seed,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                network_multiplier=network_multiplier,
                output_path=str(output_path),
                output_ext="png",
                **config_kwargs,
            )
        ], sampler="flowmatch")

    return output_path


def default_output_path(output_dir: Path, tag: str, seed: int) -> Path:
    return output_dir / f"{tag}_{seed}_{int(time.time())}.png"


def main():
    parser = argparse.ArgumentParser(description="Generate a poster with Flex.2 + Canny control.")
    parser.add_argument("--prompt", help="Text prompt")
    parser.add_argument("--canny", type=Path, required=True, help="Canny control image")
    parser.add_argument(
        "--lora",
        nargs="?",
        const="default",
        default=None,
        help="LoRA path, Hugging Face repo (user/repo), or user/repo/file.safetensors",
    )
    parser.add_argument("--output_dir", type=Path, default="./outputs")
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--lora_scale', type=float, default=1.0)
    args = parser.parse_args()

    canny_path = args.canny.expanduser()
    if not canny_path.is_file():
        raise FileNotFoundError(f"Canny image not found: {canny_path}")

    lora_path = resolve_lora_path(args.lora)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "lora" if lora_path else "base"
    output_path = default_output_path(args.output_dir, tag, args.seed)

    print("Loading Flex.2...")
    sd = load_model()
    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        attach_lora(sd, lora_path)

    generate_poster(
        sd=sd,
        prompt=args.prompt,
        canny_path=canny_path,
        output_path=output_path,
        width=args.width,
        height=args.height,
        seed=args.seed,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        network_multiplier=args.lora_scale,
    )

    print(f"Saved {output_path}")
    del sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
