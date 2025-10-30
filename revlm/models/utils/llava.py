def preprocess_llava(images, prompts, processor):
    """Build chat-template inputs for LLaVA-style VLMs with a single image and text.
    Returns CPU tensors; caller moves to device.
    """
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

