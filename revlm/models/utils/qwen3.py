def preprocess_qwen3(images, prompts, processor):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": images[0]},
            {"type": "text", "text": prompts[0]},
        ],
    }]
    return processor.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )