from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

def _resolve_model_id(config):
    model_name = getattr(getattr(config, "model", {}), "name", "")
    model_pt = getattr(getattr(config, "model", {}), "pt", None)
    return model_pt or model_name or "Salesforce/instructblip-vicuna-7b"


def get_hf_model_instructblip(config, cache_dir=None, torch_dtype=None):
    model_id = _resolve_model_id(config)
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    return InstructBlipForConditionalGeneration.from_pretrained(model_id, **load_kwargs)


def get_processor_instructblip(config, cache_dir=None):
    model_id = _resolve_model_id(config)
    return InstructBlipProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


def get_tokenizer_instructblip(config, cache_dir=None):
    return get_processor_instructblip(config, cache_dir=cache_dir).tokenizer
