import os
import logging
import transformers
from torch import nn

LOG = logging.getLogger(__name__)


def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "/scratch/jq2uw/MME/instruct_vlm_edit/ckpts/"
    os.makedirs(path, exist_ok=True)
    return path


def get_processor(config):
    """Load vision-language processor"""
    return transformers.AutoProcessor.from_pretrained(
        config.model.name,
        cache_dir=ckpt_dir(),
        trust_remote_code=True,
    )


def get_tokenizer(config):
    """Get text tokenizer from processor"""
    processor = get_processor(config)
    return processor.tokenizer


def get_model_class_for_name(model_name):
    """Get specific model class if needed, else return None for auto mode"""
    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        return getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    elif "instructblip" in model_name_lower:
        return getattr(transformers, "InstructBlipForConditionalGeneration", None)
    elif "llava" in model_name_lower:
        return getattr(transformers, "LlavaForConditionalGeneration", None)
    elif "blip2" in model_name_lower or "minigpt" in model_name_lower:
        return getattr(transformers, "Blip2ForConditionalGeneration", None)
    return None  # Use auto mode


def get_hf_model(config):
    """Load HuggingFace VLM model - auto mode"""
    ModelClass = get_model_class_for_name(config.model.name)
    if ModelClass is None:
        ModelClass = getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    
    model_path = getattr(config.model, "pt", None) or config.model.name
    # Prefer lighter dtype and auto device placement to reduce OOM risk
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    # Default to bfloat16 when available, else float16
    try:
        import torch
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else None
    except Exception:
        pass

    if not config.model.pt:
        load_kwargs["cache_dir"] = ckpt_dir()

    model = ModelClass.from_pretrained(
        model_path,
        **{k: v for k, v in load_kwargs.items() if v is not None},
    )
    
    # Set dropout if specified
    dropout = getattr(config, "dropout", None)
    if dropout is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout
            elif hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = dropout
            elif hasattr(m, "activation_dropout") and isinstance(m.activation_dropout, float):
                m.activation_dropout = dropout
    
    return model


def build_vlm(config):
    """Return (model, processor, tokenizer, inner_wrapper) with special handling by class_name or name.
    Keeps wrapper generic by centralizing selection here.
    """
    requested_class = getattr(getattr(config, "model", {}), "class_name", None)
    model_name_lower = getattr(getattr(config, "model", {}), "name", "").lower()

    if requested_class == "Qwen3VLM" or "qwen3" in model_name_lower:
        # Lazy import to avoid hard dependency at module import time
        from .qwen3 import Qwen3VLM
        inner = Qwen3VLM(config)
        return inner.model, inner.processor, getattr(inner, "tokenizer", None), inner

    # Default path
    model = get_hf_model(config)
    processor = get_processor(config)
    tokenizer = get_tokenizer(config)
    return model, processor, tokenizer, None

