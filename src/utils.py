def fix_prompt_length(tokenizer, prompts):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    return tokenizer.batch_decode(text_inputs.input_ids, skip_special_tokens=True)
