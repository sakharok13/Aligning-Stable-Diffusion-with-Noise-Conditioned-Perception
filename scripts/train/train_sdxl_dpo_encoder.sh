# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_diffusion_dpo_SDXL.py \
python src/train/train_SDXL.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
 --output_dir="dpo-encoder-sdxl" \
 --mixed_precision="fp16" \
 --dataset_name="yuvalkirstain/pickapic_v2"\
 --cache_dir="~/Gambashidze/pickapic_v2" \
 --resolution=1024 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=64 \
 --gradient_checkpointing \
 --rank=64 \
 --learning_rate=1e-8 \
 --scale_lr \
 --report_to="wandb" \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=500 \
 --max_train_steps=2000 \
 --checkpointing_steps=50 \
 --seed=69 \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --dataloader_num_workers=4 \
 --vae_encode_batch_size=8 \
 --run_validation \
 --validation_steps=50 \
 --beta_dpo=2500 \
 --method="encoder"
