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


def get_hf_model(config):
    """Load HuggingFace VLM model - auto mode"""
    from .adapters import get_model_class_for_name
    
    # Try specific class first, then fall back to auto
    ModelClass = get_model_class_for_name(config.model.name)
    if ModelClass is None:
        ModelClass = getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    
    model_path = getattr(config.model, "pt", None) or config.model.name
    model = ModelClass.from_pretrained(
        model_path,
        cache_dir=ckpt_dir() if not config.model.pt else None,
        trust_remote_code=True,
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

def get_model_class_for_name(model_name):
    """Get specific model class if needed, else return None for auto mode"""
    model_name_lower = model_name.lower()
    
    # Only add specific classes if AutoModel doesn't work
    if "qwen3" in model_name_lower:
        return getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    elif "instructblip" in model_name_lower:
        return getattr(transformers, "InstructBlipForConditionalGeneration", None)
    elif "llava" in model_name_lower:
        return getattr(transformers, "LlavaForConditionalGeneration", None)
    elif "blip2" in model_name_lower or "minigpt" in model_name_lower:
        return getattr(transformers, "Blip2ForConditionalGeneration", None)
    
    return None  # Use auto mode

