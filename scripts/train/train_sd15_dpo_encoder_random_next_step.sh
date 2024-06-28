# CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train_sd15.py \
# python -m pdb src/train/train_sd15.py \
python src/train/train_sd15.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
 --output_dir="encoder-warmup-random-ts" \
 --mixed_precision="fp16" \
 --dataset_name="yuvalkirstain/pickapic_v2"\
 --cache_dir="~/Gambashidze/pickapic_v2" \
 --dataset_name="/home/jovyan/Gambashidze/SD-DPO/abs_winners_pick.csv" \
 --method="encoder" \
 --is_csv_dataset=True \
 --resolution=512 \
 --train_batch_size=64 \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --rank=64 \
 --learning_rate=5e-7 \
 --scale_lr \
 --report_to="wandb" \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=250 \
 --max_train_steps=1000 \
 --checkpointing_steps=100 \
 --seed=1 \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --dataloader_num_workers=4 \
 --vae_encode_batch_size=32 \
 --run_validation \
 --validation_steps=100 \
 --beta_dpo=2500 \
 --use_middle_states \
 --collect_intermediate_steps=False \
 --timestep="uniform"
 # --use_upsample_states_num=2
 # --learning_rate=1e-8 \
