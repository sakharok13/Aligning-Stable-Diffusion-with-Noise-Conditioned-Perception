import torch
from transformers import PretrainedConfig


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_captions(tokenizer, examples):
    captions = []
    for caption in examples["caption"]:
        captions.append(caption)

    text_inputs = tokenizer(
        captions,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    return text_inputs.input_ids


@torch.no_grad()
def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = None

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def encode_vae_batched(vae, feed_pixel_values, args):
    latents = []
    for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
        latents.append(
            vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
        )
    latents = torch.cat(latents, dim=0)
    return latents


def decode_vae_batched(vae, feed_pixel_values, args):
    latents = []
    for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
        latents.append(
            vae.decode(feed_pixel_values[i : i + args.vae_encode_batch_size]).sample
        )
    latents = torch.cat(latents, dim=0)
    return latents