#!/usr/bin/env python3
"""Gradio UI for Flex.2 poster generation."""

import argparse
import gc
import threading
import time
from pathlib import Path

import gradio as gr
import torch

from infer import attach_lora, generate_poster, load_model, resolve_lora_path

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs" / "ui"

MODEL = None
MODEL_LOCK = threading.Lock()
LORA_LABEL = "base"


def normalize_size(value: int) -> int:
    value = int(value)
    value = max(256, min(2048, value))
    return value - (value % 16)


def load_runtime(lora_arg):
    global MODEL, LORA_LABEL

    print("Loading Flex.2...")
    MODEL = load_model()

    lora_path = resolve_lora_path(lora_arg)
    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        attach_lora(MODEL, lora_path)
        LORA_LABEL = "lora"
    else:
        LORA_LABEL = "base"


def run_generation(
    prompt,
    canny_path,
    width,
    height,
    seed,
    guidance_scale,
    steps,
    lora_scale,
):
    if MODEL is None:
        raise gr.Error("Model is not loaded yet.")
    prompt = (prompt or "").strip()
    if not prompt and not canny_path:
        raise gr.Error("Please enter a prompt or upload a Canny control image.")

    width = normalize_size(width)
    height = normalize_size(height)
    seed = int(seed)
    steps = int(steps)
    guidance_scale = float(guidance_scale)
    lora_scale = float(lora_scale)

    output_path = OUTPUT_DIR / f"{LORA_LABEL}_{seed}_{int(time.time())}.png"

    with MODEL_LOCK:
        result_path = generate_poster(
            sd=MODEL,
            prompt=prompt,
            canny_path=Path(canny_path) if canny_path else None,
            output_path=output_path,
            width=width,
            height=height,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            network_multiplier=lora_scale,
        )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(result_path), f"Saved: {result_path}"


def build_ui():
    with gr.Blocks(title="Flex.2 Poster Generator") as demo:
        gr.Markdown("# Flex.2 Poster Generator")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt (optional)",
                    lines=4,
                    placeholder="minimalist book sale poster with bold typography",
                )
                canny = gr.Image(label="Canny Control Image (optional)", type="filepath")

                with gr.Row():
                    width = gr.Number(label="Width", value=768, precision=0)
                    height = gr.Number(label="Height", value=768, precision=0)

                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    guidance = gr.Slider(label="Guidance", minimum=0.0, maximum=12.0, value=4.0, step=0.1)

                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=80, value=25, step=1)
                    lora_scale = gr.Slider(label="LoRA Scale", minimum=0.0, maximum=2.0, value=1.0, step=0.05)

                generate = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                result = gr.Image(label="Generated Poster", type="filepath")
                status = gr.Textbox(label="Output", interactive=False)

        generate.click(
            fn=run_generation,
            inputs=[prompt, canny, width, height, seed, guidance, steps, lora_scale],
            outputs=[result, status],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Flex.2 poster Gradio UI.")
    parser.add_argument(
        "--lora",
        nargs="?",
        const="default",
        default="default",
        help="LoRA path, Hugging Face repo, or omit the value to use the default LoRA.",
    )
    parser.add_argument("--no-lora", action="store_true", help="Launch with the base Flex.2 model only.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    load_runtime(None if args.no_lora else args.lora)
    demo = build_ui()
    demo.queue(max_size=8).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
