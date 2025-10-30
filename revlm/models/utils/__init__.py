import os
import logging
import transformers
from torch import nn
import torch

from .qwen3 import *
from .llava import *

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
        return getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    elif "instructblip" in model_name_lower:
        return getattr(transformers, "InstructBlipForConditionalGeneration", None)
    elif "llava" in model_name_lower:
        return getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    elif "blip2" in model_name_lower or "minigpt" in model_name_lower:
        return getattr(transformers, "Blip2ForConditionalGeneration", None)
    return None  # Use auto mode


def get_hf_model(config):
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()
    ModelClass = get_model_class_for_name(name_lower)
    model_path = getattr(config.model, "pt", None) or config.model.name
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else None
    if not config.model.pt:
        load_kwargs["cache_dir"] = ckpt_dir()
    model = ModelClass.from_pretrained(
        model_path,
        **{k: v for k, v in load_kwargs.items() if v is not None},
    )
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


def get_preprocess(config):
    """Return a callable(images, prompts, processor) -> dict of tensors on CPU.
    Wrapper will move tensors to device.
    """
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()

    def _generic(images, prompts, processor):
        return processor(images=images, text=prompts, return_tensors="pt", padding=True)

    if "qwen3" in name_lower:
        return preprocess_qwen3
    if "llava" in name_lower or "onevision" in name_lower:
        return preprocess_llava
    return _generic

