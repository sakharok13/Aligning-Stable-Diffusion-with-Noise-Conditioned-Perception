import argparse
import math
from pathlib import Path
from more_itertools import chunked as batched
from tqdm import tqdm

import torch
from accelerate import Accelerator
from diffusers import DiffusionPipeline, UNet2DConditionModel

from prompts import get_prompts
from utils import fix_prompt_length


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple non-parallel batched SD inference script for prompt benchmarks.")
    parser.add_argument(
        "--save_images_path",
        type=str,
        default=None,
        required=True,
        help="Folder path for images created by inference.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--replacement_unet",
        type=str,
        default=None,
        required=False,
        help="Path to pretraind unet to swap into pipe.",
    )
    parser.add_argument(
        "--lora_unet",
        type=str,
        default=None,
        required=False,
        help="Path to pretraind unet lora to swap into pipe.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="parti",
        required=False,
        choices=["hps", "parti", "pick"],
        help="Run on benchmark prompts supported by prompts.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="Number of prompts to send to pipe at the same time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        required=False,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        required=False,
        help="Classifier-free guidance scale parameter.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    prompts = get_prompts(args.prompts)

    accelerator = Accelerator()
    pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    if args.replacement_unet is not None:
        pipe.unet = UNet2DConditionModel.from_pretrained(args.replacement_unet, subfolder="unet")
    if args.lora_unet is not None:
        pipe.load_lora_weights(args.lora_unet)
    pipe.safety_checker = None
    pipe.enable_vae_slicing()
    pipe = pipe.to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images_path = Path(args.save_images_path) / args.prompts
    images_path.mkdir(parents=True, exist_ok=True)

    with accelerator.split_between_processes(prompts) as prompts_split:
        data_loader = batched(prompts_split, args.batch_size)
        total = math.ceil(len(prompts_split)/args.batch_size)
        for batch_index, prompt_batch in tqdm(enumerate(data_loader), total=total):
            prompt_batch = fix_prompt_length(pipe.tokenizer, prompt_batch)
            image_batch = pipe(
                prompt_batch,
                guidance_scale=args.guidance_scale,
                generator=[generator]*len(prompt_batch)
            )
            for i, image in enumerate(image_batch.images):
                process_start_index = math.ceil(len(prompts) / accelerator.num_processes) * accelerator.process_index
                image_index = process_start_index + batch_index * args.batch_size + i
                image.save(images_path / f"{image_index:05d}.jpg")


if __name__ == "__main__":
    args = parse_args()
    main(args)
