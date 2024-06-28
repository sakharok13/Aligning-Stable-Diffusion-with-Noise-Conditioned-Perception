#!/bin/bash

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" accelerate launch src/inference.py \
    --save_images_path test_images/dpo-sdxl_abs_encoder_50 \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --batch_size 32 \
    --prompts pick \
    --seed 1 \
    --lora_unet dpo-encoder-sdxl_abs/checkpoint-50/pytorch_lora_weights.safetensors \


