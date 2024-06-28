
# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch src/train/train_sd15.py \
python src/train/train_sd15.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
 --output_dir="sd15_ddpo_abs_winners_256_orig" \
 --mixed_precision="fp16" \
 --dataset_name="yuvalkirstain/pickapic_v2"\
 --cache_dir="~/Gambashidze/pickapic_v2" \
 --report_to="wandb" \
 --resolution=512 \
 --train_batch_size=64 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --rank=64 \
 --learning_rate=8e-8 \
 --scale_lr \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=60 \
 --max_train_steps=600 \
 --checkpointing_steps=10 \
 --seed=1 \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --dataloader_num_workers=4 \
 --vae_encode_batch_size=32 \
 --run_validation \
 --validation_steps=20 \
 --beta_dpo=2000 \
 --method="dpo"