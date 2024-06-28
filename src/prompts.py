import json
import pandas as pd
import huggingface_hub


def get_hps_prompts():
    hps_prompt_files = ['anime.json', 'concept-art.json', 'paintings.json', 'photo.json']
    all_prompts = []
    for file in hps_prompt_files:
        file_name = huggingface_hub.hf_hub_download("zhwang/HPDv2", file, subfolder="benchmark", repo_type="dataset")
        with open(file_name) as f:
            prompts = json.load(f)
            all_prompts.extend(prompts)
    return all_prompts


def get_parti_prompts():
    file_name = huggingface_hub.hf_hub_download("nateraw/parti-prompts", "PartiPrompts.tsv", repo_type="dataset")
    return pd.read_csv(file_name, sep="\t")["Prompt"].to_list()


def get_prompts(prompts_name):
    if prompts_name == "hps":
        prompts = get_hps_prompts()
    elif prompts_name == "parti":
        prompts = get_parti_prompts()
    elif prompts_name == "pick":
        prompts = pd.read_csv('pick_val_prompts.csv').prompt.tolist()
    else:
        raise ValueError()
    return prompts
