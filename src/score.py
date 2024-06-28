import argparse
import math
from pathlib import Path

import pandas as pd
from more_itertools import chunked as batched
from more_itertools import unzip
from PIL import Image
from tqdm import tqdm

import huggingface_hub
import torch
import torch.nn as nn
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import hps_version_map
from ImageReward import ImageReward
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from prompts import get_prompts


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple non-parallel non-batched script for evaluating HPSv2 model on a folder of images.")
    parser.add_argument(
        "--images_path",
        type=str,
        default=None,
        required=True,
        help="Folder path with images corresponding to a benchmark.",
    )
    parser.add_argument(
        "--save_scores_path",
        type=str,
        default=None,
        required=True,
        help="Folder path to save csv with scores to.",
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
        help="Number of prompt-image pairs to send to model at the same time.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        required=False,
        choices=["all"],
        help="Models to use in scoring.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


class HPSv2Model:
    def __init__(self, device, hps_version="v2.1"):
        self.device = device

        model, _, preprocessor = create_model_and_transforms(
            'ViT-H-14',
            # 'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        tokenizer = get_tokenizer('ViT-H-14')
        checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval().requires_grad_(False)

        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def score(self, image, prompt):
        image = self.preprocessor(image).unsqueeze(0).to(device=self.device)
        text = self.tokenizer([prompt]).to(device=self.device)

        with torch.cuda.amp.autocast():
            outputs = self.model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            return logits_per_image.cpu().numpy().item()

    def score_batch(self, images, prompts):
        images = torch.stack([self.preprocessor(image) for image in images], dim=0).to(device=self.device)
        prompts = self.tokenizer(prompts).to(device=self.device)

        with torch.cuda.amp.autocast():
            outputs = self.model(images, prompts)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = (image_features * text_features).sum(-1)
            return logits_per_image.flatten().cpu().numpy().tolist()


class PickScoreModel:
    def __init__(self, device):
        self.device = device

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        processor = AutoProcessor.from_pretrained(processor_name_or_path)
        model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
        model.requires_grad_(False)

        self.model = model
        self.processor = processor

    def score(self, image, prompt):
        image_inputs = self.processor(
            images=[image],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_inputs = self.processor(
            text=[prompt],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        return scores.cpu().numpy().item()

    def score_batch(self, images, prompts):
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(-1)
        return scores.flatten().cpu().numpy().tolist()


class ImageRewardModel:
    def __init__(self, device):
        self.device = device

        checkpoint_path = huggingface_hub.hf_hub_download("THUDM/ImageReward", filename="ImageReward.pt")
        config_path = huggingface_hub.hf_hub_download("THUDM/ImageReward", filename="med_config.json")
        model = ImageReward(device=device, med_config=config_path).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval().requires_grad_(False)

        self.model = model

    def score(self, image, prompt):
        image = self.model.preprocess(image).unsqueeze(0).to(self.device)
        text = self.model.blip.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt"
        ).to(self.device)

        imagereward_score = self.model.score_gard(text.input_ids, text.attention_mask, image)
        return imagereward_score.cpu().numpy().item()

    def score_batch(self, images, prompts):
        image_inputs = torch.stack([self.model.preprocess(image) for image in images]).to(self.device)
        text_inputs = self.model.blip.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt"
        ).to(self.device)

        imagereward_score = self.model.score_gard(text_inputs.input_ids, text_inputs.attention_mask, image_inputs)
        return imagereward_score.flatten().cpu().numpy().tolist()


class CLIPScoreModel:
    def __init__(self, device, clip=None):
        self.device = device

        if clip is None:
            clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip = clip.to(device)
        clip.eval().requires_grad_(False)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.clip = clip
        self.processor = processor

    def score(self, image, prompt):
        inputs = self.processor(prompt, image, truncation=True, max_length=77, return_tensors="pt").to(self.device)

        embeds = self.clip(**inputs)
        clip_score = torch.sum(torch.mul(embeds.text_embeds, embeds.image_embeds))
        return clip_score.cpu().numpy().item()

    def score_batch(self, images, prompts):
        inputs = self.processor(
            prompts, 
            images, 
            truncation=True, 
            padding=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(self.device)

        embeds = self.clip(**inputs)
        clip_score = (embeds.text_embeds * embeds.image_embeds).sum(-1)
        return clip_score.flatten().cpu().numpy().tolist()


class AestheticModel:
    def __init__(self, device, clip=None):
        self.device = device

        if clip is None:
            clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip = clip.to(device)
        clip.eval().requires_grad_(False)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        checkpoint_path = huggingface_hub.hf_hub_download(
            "camenduru/improved-aesthetic-predictor", filename="sac+logos+ava1-l14-linearMSE.pth")
        model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for key in list(checkpoint.keys()):
            checkpoint[key.lstrip("layers.")] = checkpoint.pop(key)
        model.load_state_dict(checkpoint)
        model.eval().requires_grad_(False)

        self.clip = clip
        self.processor = processor
        self.model = model

    def score(self, image, prompt):
        inputs = self.processor(prompt, image, truncation=True, max_length=77, return_tensors="pt").to(self.device)

        embeds = self.clip(**inputs)
        aesthetic_score = self.model(embeds.image_embeds)

        return aesthetic_score.cpu().numpy().item()

    def score_batch(self, images, prompts):
        inputs = self.processor(
            prompts, 
            images, 
            truncation=True, 
            padding=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(self.device)

        embeds = self.clip(**inputs)
        aesthetic_score = self.model(embeds.image_embeds)

        return aesthetic_score.flatten().cpu().numpy().tolist()


def create_models_from_args(args):
    device = "cuda"
    if args.models == "all":
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        return {
            "hpsv2": HPSv2Model(device),
            "pick": PickScoreModel(device),
            "imagereward": ImageRewardModel(device),
            "clip": CLIPScoreModel(device, clip),
            "aesthetics": AestheticModel(device, clip)
        }
    else:
        raise NotImplementedError()


def main(args):
    image_paths = sorted((Path(args.images_path) / args.prompts).glob("*.jpg"))
    prompts = get_prompts(args.prompts)

    models_collection = create_models_from_args(args)
    scores_collection = {k: [] for k in models_collection.keys()}

    if args.batch_size == 1:
        for image_path, prompt in tqdm(zip(image_paths, prompts), total=len(prompts)):
            image = Image.open(image_path)
    
            for model in scores_collection.keys():
                score = models_collection[model].score(image, prompt)
                scores_collection[model].append(score)
    else:
        data_loader = batched(zip(image_paths, prompts), args.batch_size)
        for batch in tqdm(data_loader, total=math.ceil(len(prompts)/args.batch_size)):
            image_paths, prompts = unzip(batch)
            images = [Image.open(image_path) for image_path in image_paths]
            prompts = list(prompts)
    
            for model in scores_collection.keys():
                scores = models_collection[model].score_batch(images, prompts)
                scores_collection[model].extend(scores)

    save_scores_path = Path(args.save_scores_path) / args.prompts
    save_scores_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores_collection).to_csv(save_scores_path / "scores.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
