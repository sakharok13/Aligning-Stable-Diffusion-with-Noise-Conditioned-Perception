#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from copy import deepcopy
import io
import logging
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import pandas as pd

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from model_utils import get_unet_middle_states
from scheduler_utils import ddpm_scheduler_step_batched, ddpm_scheduler_step_to_orig_batched
from validation_utils import log_validation
from utils import (
    import_model_class_from_model_name_or_path,
    tokenize_captions,
    encode_prompt,
    encode_vae_batched,
    decode_vae_batched
)

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--is_csv_dataset",
        type=bool,
        default=False,
        help=(
            "Whether the dataset is from csv."
        ),
    )
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="train",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=False,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=2500,
        help="DPO KL Divergence penalty.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        help="DPO loss type. Can be one of 'sigmoid' (default), 'ipo', or 'cpo'",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-dpo-lora",
        help=("The name of the tracker to report results to."),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dpo",
        # choices=["dpo", "encoder", "both"],
        help=("Enable our fancy method."),
    )
    parser.add_argument(
        "--encoder_loss_coeff",
        type=float,
        default=1.0,
        help=("Coefficient for encoder loss. Works for both 'encoder' and 'both'."),
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained CLIP model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pool_encoder",
        action="store_true",
        help="Pool spatial dimensions of Unet before using in loss.",
    )
    parser.add_argument(
        "--use_middle_states",
        action="store_true",
        help="Use Unet states after mid block rather than after downsample block.",
    )
    parser.add_argument(
        "--use_upsample_states_num",
        type=int,
        default=None,
        help=("Number of unet upsample blocks to use."),
    )
    parser.add_argument(
        "--collect_intermediate_steps",
        type=bool,
        default=False,
        help=("Whether we need to collect intermediate states of encoder."),
    )
    parser.add_argument(
        "--normalize_intermediate_steps",
        type=bool,
        default=False,
        help=("Whether we need to normalize intermediate states of encoder."),
    )
    parser.add_argument(
        "--use_current_unet",
        action="store_true",
        help="Use current Unet as encoder (with no grad), as opposed to reference by default.",
    )
    parser.add_argument(
        "--timestep",
        type=str,
        default="t-1",
        help="Timestep to bring x0_pred to when embedding it into U-Net for perceptual loss.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    prediction_type = 'epsilon' # if args.timestep == 't-1' else 'v_prediction'
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        prediction_type=prediction_type,
        subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )
    if args.method in ["encoder", "both"]:
        encoder = deepcopy(unet)
    elif args.method in ["clip"]:
        from kornia.geometry.transform import Resize
        from kornia.augmentation import Denormalize, Normalize

        encoder = CLIPVisionModelWithProjection.from_pretrained(args.clip_model_name_or_path)
        processor = CLIPImageProcessor.from_pretrained(args.clip_model_name_or_path)
        # differentiable preprocessing for CLIP: resize, undo sd normalization, apply clip normalization
        h, w = processor.crop_size["height"], processor.crop_size["width"]
        clip_transform = nn.Sequential(
            Resize((h, w), interpolation="bicubic", antialias=False),
            Denormalize(mean=0.5, std=0.5),
            Normalize(mean=processor.image_mean, std=processor.image_std)
        )
    else:
        encoder = nn.Identity()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    encoder.to(accelerator.device, dtype=weight_dtype)

    # Set up LoRA.
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = LoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            cast_training_params(unet_, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if args.is_csv_dataset:
        train_dataset = Dataset.from_pandas(pd.read_csv(args.dataset_name))
    else:
        train_dataset = load_dataset(
            args.dataset_name,
            cache_dir=args.cache_dir,
            split=args.dataset_split_name,
        )
    

    train_transforms = transforms.Compose(
        [
            transforms.Resize(int(args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    def find_absolute_winners(examples):
        df = examples
        grouped = df.groupby('caption')
        absolute_winners = []
        for prompt, group in grouped:
            wins = defaultdict(int)
            losses = defaultdict(int)
            for _, row in group.iterrows():
                img_0 = row['image_0_uid']
                img_1 = row['image_1_uid']
                label_0 = row['label_0']
                label_1 = row['label_1']
                
                if label_0 == 0:
                    losses[img_0] += 1
                    wins[img_1] += 1
                else:
                    losses[img_1] += 1
                    wins[img_0] += 1
    
                if label_1 == 0:
                    losses[img_1] += 1
                    wins[img_0] += 1
                else:
                    losses[img_0] += 1
                    wins[img_1] += 1
            potential_winners = [img for img in wins if wins[img] > 0 and losses[img] == 0]
            for _, row in group.iterrows():
                if row['image_0_uid'] in potential_winners or row['image_1_uid'] in potential_winners:
                    absolute_winners.append(row)
                    if row['image_0_uid'] in potential_winners:
                        potential_winners.remove(row['image_0_uid'])
                    if row['image_1_uid'] in potential_winners:
                        potential_winners.remove(row['image_1_uid'])
                    if not potential_winners:
                        break
        
        return pd.DataFrame(absolute_winners).to_dict('list')
    
    def preprocess_train(examples):
        all_pixel_values = []
        # for col_name in ["jpg_0", "jpg_1"]:
        for col_name in ['image_0_uid',	'image_1_uid']:
            images = [Image.open('/home/jovyan/Gambashidze/train/' + uid + '.jpg').convert("RGB") for uid in examples[col_name]]
            pixel_values = [train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)
        examples["pixel_values"] = combined_pixel_values

        examples["input_ids"] = tokenize_captions(tokenizer, examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        final_dict = {"pixel_values": pixel_values}
        final_dict["input_ids"] = torch.stack([example["input_ids"] for example in examples])
        return final_dict

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    unet.train()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                need_encoder = args.method in ["encoder", "both"]
                need_default = args.method in ["dpo", "both"]
                need_clip = args.method == "clip"
                need_ref = args.loss_type in ["sigmoid", "hinge", "ipo"]
                need_timestep = args.timestep if not need_clip else "0"

                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))

                latents = encode_vae_batched(vae, feed_pixel_values, args)
                latents = latents * vae.config.scaling_factor

                bs, c, h, w = latents.shape
                bs = bs // 2

                # Sample noise that we'll add to the latents
                noise = torch.randn((bs, c, h, w), dtype=latents.dtype, device=latents.device).repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), dtype=torch.long, device=latents.device
                ).repeat(2)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                prompt_embeddings = encode_prompt(text_encoder, batch["input_ids"]).repeat(2, 1, 1)

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeddings,
                ).sample

                # Reference model predictions.
                if need_ref:
                    accelerator.unwrap_model(unet).disable_adapters()
                    with torch.no_grad():
                        ref_noise_pred = unet(
                            noisy_latents,
                            timesteps,
                            prompt_embeddings,
                        ).sample.detach()
                    # Re-enable adapters.
                    accelerator.unwrap_model(unet).enable_adapters()

                if (need_encoder or need_clip):
                    if need_timestep == "t-1":
                        # Previous timestep to use in encoder unet
                        timesteps_prev = timesteps - 1
                        timesteps_prev[timesteps_prev < 0] = 0

                        # Compute x_{t-1} latents using predicted and actual noise
                        variance_noise = torch.randn(
                            (bs, c, h, w), dtype=latents.dtype, device=latents.device
                        ).repeat(2, 1, 1, 1)
                        noisy_latents_prev_pred = ddpm_scheduler_step_batched(
                            noise_scheduler,
                            noise_pred,
                            timesteps,
                            timesteps_prev,
                            noisy_latents,
                            variance_noise
                        ).prev_sample.to(latents.dtype)
                        if need_ref:
                            ref_noisy_latents_prev_pred = ddpm_scheduler_step_batched(
                                noise_scheduler,
                                ref_noise_pred,
                                timesteps,
                                timesteps_prev,
                                noisy_latents,
                                variance_noise
                            ).prev_sample.to(latents.dtype)
                        noisy_latents_prev = ddpm_scheduler_step_batched(
                            noise_scheduler,
                            noise,
                            timesteps,
                            timesteps_prev,
                            noisy_latents,
                            variance_noise
                        ).prev_sample.to(latents.dtype)

                    elif need_timestep == "uniform":
                        # Get timestep_0 prediction
                        noisy_latents_zero_pred, original_noise_pred = ddpm_scheduler_step_to_orig_batched(
                            noise_scheduler,
                            noise_pred,
                            timesteps,
                            noisy_latents
                        )
                        noisy_latents_zero_pred = noisy_latents_zero_pred.to(latents.dtype)
                        original_noise_pred = original_noise_pred.to(latents.dtype)
                        if need_ref:
                            ref_noisy_latents_zero_pred, ref_original_noise_pred = ddpm_scheduler_step_to_orig_batched(
                                noise_scheduler,
                                ref_noise_pred,
                                timesteps,
                                noisy_latents
                            )
                            ref_noisy_latents_zero_pred = ref_noisy_latents_zero_pred.to(latents.dtype)
                            ref_original_noise_pred = ref_original_noise_pred.to(latents.dtype)

                        timesteps_prev = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps, (bs,), dtype=torch.long, device=latents.device
                        ).repeat(2)
                        
                        noisy_latents_prev_pred = noise_scheduler.add_noise(
                            noisy_latents_zero_pred,
                            original_noise_pred,
                            timesteps_prev,
                        ).to(latents.dtype)
                        if need_ref:
                            ref_noisy_latents_prev_pred = noise_scheduler.add_noise(
                                ref_noisy_latents_zero_pred,
                                ref_original_noise_pred,
                                timesteps_prev,
                            ).to(latents.dtype)
                        noisy_latents_prev = noise_scheduler.add_noise(
                            latents,
                            noise,
                            timesteps_prev,
                        ).to(latents.dtype)
                    else:
                        raise NotImplementedError()

                if need_encoder:
                    if args.use_current_unet:
                        encoder = deepcopy(accelerator.unwrap_model(unet))
                        encoder.requires_grad_(False)
                    # Encode predicted samples with unet middle states
                    encoded_pred = get_unet_middle_states(
                        encoder,
                        noisy_latents_prev_pred,
                        timesteps_prev,
                        prompt_embeddings,
                        use_middle_states=args.use_middle_states,
                        use_upsample_states_num=args.use_upsample_states_num,
                        collect_intermediate_steps=args.collect_intermediate_steps,
                        normalize_intermediate_steps=args.normalize_intermediate_steps
                    )
                    if need_ref:
                        ref_encoded_pred = get_unet_middle_states(
                            encoder,
                            ref_noisy_latents_prev_pred,
                            timesteps_prev,
                            prompt_embeddings,
                            use_middle_states=args.use_middle_states,
                            use_upsample_states_num=args.use_upsample_states_num,
                            collect_intermediate_steps=args.collect_intermediate_steps,
                            normalize_intermediate_steps=args.normalize_intermediate_steps
                        )
                    encoded_target = get_unet_middle_states(
                        encoder,
                        noisy_latents_prev,
                        timesteps_prev,
                        prompt_embeddings,
                        use_middle_states=args.use_middle_states,
                        use_upsample_states_num=args.use_upsample_states_num,
                        collect_intermediate_steps=args.collect_intermediate_steps,
                        normalize_intermediate_steps=args.normalize_intermediate_steps
                    )
                    if args.pool_encoder:
                        encoded_pred = encoded_pred.mean(dim=(2,3), keepdim=True)
                        ref_encoded_pred = ref_encoded_pred.mean(dim=(2,3), keepdim=True)
                        encoded_target = encoded_target.mean(dim=(2,3), keepdim=True)

                elif need_clip:
                    # Process x_0 latents through vae, resize and normalize, and feed to clip
                    decoded_images_pred = decode_vae_batched(vae, noisy_latents_prev_pred, args)
                    decoded_images_pred = clip_transform(decoded_images_pred)
                    encoded_pred = encoder(decoded_images_pred).image_embeds[..., None, None]

                    if need_ref:
                        ref_decoded_images_pred = decode_vae_batched(vae, ref_noisy_latents_prev_pred, args)
                        ref_decoded_images_pred = clip_transform(ref_decoded_images_pred)
                        ref_encoded_pred = encoder(ref_decoded_images_pred).image_embeds[..., None, None]

                    decoded_images = feed_pixel_values
                    decoded_images = clip_transform(decoded_images)
                    encoded_target = encoder(decoded_images).image_embeds[..., None, None]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute losses.
                if need_encoder and not need_default:
                    model_losses = (encoded_pred.float() - encoded_target.float()).pow(2).mean(dim=(1, 2, 3))
                    model_losses = model_losses * args.encoder_loss_coeff
                    if need_ref:
                        ref_losses = (ref_encoded_pred.float() - encoded_target.float()).pow(2).mean(dim=(1, 2, 3))
                        ref_losses = ref_losses * args.encoder_loss_coeff

                elif need_encoder and need_default:
                    encoder_losses = (encoded_pred.float() - encoded_target.float()).pow(2).mean(dim=(1, 2, 3))
                    default_losses = (noise_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
                    model_losses = default_losses + args.encoder_loss_coeff * encoder_losses
                    if need_ref:
                        ref_encoder_losses = (ref_encoded_pred.float() - encoded_target.float()).pow(2).mean(dim=(1, 2, 3))
                        ref_default_losses = (ref_noise_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
                        ref_losses = ref_default_losses + args.encoder_loss_coeff * ref_encoder_losses

                elif need_default and not need_encoder:
                    model_losses = (noise_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
                    if need_ref:
                        ref_losses = (ref_noise_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))

                model_losses_w, model_losses_l = model_losses.chunk(2)
                raw_model_loss = model_losses.mean()
                model_diff = model_losses_w - model_losses_l

                if need_ref:
                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                    raw_ref_loss = ref_losses.mean()
                    ref_diff = ref_losses_w - ref_losses_l
                else:
                    ref_diff = ref_losses_w = ref_losses_l = 0

                # Final loss
                logits = ref_diff - model_diff
                if args.loss_type == "sigmoid":
                    loss = -1 * F.logsigmoid(args.beta_dpo * logits).mean()
                elif args.loss_type == "hinge":
                    loss = torch.relu(1 - args.beta_dpo * logits).mean()
                elif args.loss_type == "ipo":
                    losses = (logits - 1 / (2 * args.beta)) ** 2
                    loss = losses.mean()
                elif args.loss_type == "cpo":
                    loss = -1 * F.logsigmoid(-args.beta_dpo * model_diff).mean()
                elif args.loss_type == "sft":
                    loss = model_losses_w.mean()
                elif args.loss_type == "orpo":
                    ratio_losses = args.beta_orpo * F.logsigmoid(model_diff)
                    loss = model_losses_w.mean() - ratio_losses.mean()
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")

                implicit_acc = (logits > 0).sum().float() / logits.size(0)
                implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)

                latent_reward_w = args.beta_dpo * (model_losses_w - ref_losses_w).mean()
                latent_reward_l = args.beta_dpo * (model_losses_l - ref_losses_l).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.run_validation and global_step % args.validation_steps == 0:
                        log_validation(
                            args,
                            unet=unet,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            epoch=epoch,
                        )

            logs = {
                "loss": loss.detach().item() + F.logsigmoid(torch.tensor(0.0)).item(),
                "model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "reward_w": latent_reward_w.detach().item(),
                "reward_l": latent_reward_l.detach().item(),
                "acc": implicit_acc.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=None
        )

        # Final validation?
        if args.run_validation:
            log_validation(
                args,
                unet=None,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=epoch,
                is_final_validation=True,
            )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
