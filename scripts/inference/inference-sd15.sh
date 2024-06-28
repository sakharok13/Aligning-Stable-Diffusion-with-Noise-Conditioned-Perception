#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

for i in $(seq 200 200 2000); do
    accelerate launch src/inference.py \
        --save_images_path "test_images/sd15_dpo_$i" \
        --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
        --lora_unet "diffusion-dpo-final/checkpoint-$i/pytorch_lora_weights.safetensors" \
        --batch_size 128 \
        --prompts pick
done
