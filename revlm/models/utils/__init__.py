import os
import logging
import transformers
from torch import nn
import torch

from .qwen3 import *
from .llava import *
from .instructblip import *

LOG = logging.getLogger(__name__)


def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "/scratch/jq2uw/MME/instruct_vlm_edit/ckpts/"
    os.makedirs(path, exist_ok=True)
    return path


def get_processor(config):
    """Load vision-language processor"""
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()
    if "blip" in name_lower:
        return get_processor_instructblip(config, cache_dir=ckpt_dir())
    
    return transformers.AutoProcessor.from_pretrained(
        config.model.name,
        cache_dir=ckpt_dir(),
        trust_remote_code=True,
    )



def get_tokenizer(config):
    """Get text tokenizer from processor"""
    processor = get_processor(config)
    tok = processor.tokenizer
    return tok


def get_model_class_for_name(model_name):
    """Get specific model class if needed, else return None for auto mode"""
    model_name_lower = model_name.lower()
    if "qwen3" in model_name_lower:
        return getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    elif "blip" in model_name_lower:
        return getattr(transformers, "InstructBlipForConditionalGeneration", None)
    elif "llava" in model_name_lower:
        return getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
    return None  # Use auto mode


def get_hf_model(config):
    name_lower = getattr(getattr(config, "model", {}), "name", "").lower()
    if "blip" in name_lower:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        cache = None if getattr(config.model, "pt", None) else ckpt_dir()
        return get_hf_model_instructblip(config, cache_dir=cache, torch_dtype=torch_dtype)
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

    def _generic(images, prompts, processor, tokenize=False):
        return processor(images=images, text=prompts, return_tensors="pt", padding=True, tokenize=tokenize)

    if "qwen3" in name_lower:
        return preprocess_qwen3
    if "llava" in name_lower:
        return preprocess_llava
    return _generic


def clean_answer(o, i):
    s = o or ""
    if "ASSISTANT:" in s:
        s = s.split("ASSISTANT:")[-1].strip()
    elif "assistant" in s:
        s = s.split("assistant")[-1].replace("\n", "").replace(":", "").strip()
    elif "Answer:" in s:
        s = s.split("Answer:")[-1].strip()
    elif "ANSWER:" in s:
        s = s.split("ANSWER:")[-1].strip()
    elif "answer:" in s:
        s = s.split("answer:")[-1].strip()
    else:
        # remove i from s
        s = s.replace(i, "").strip()
    return s


def compute_loss_stats(model, prompt_inputs, labels_ids, mask_prompt: bool = True):
    """Compute (avg_nll, sum_nll, num_tokens) for given labels given prompt inputs.
    Handles both encoder-decoder and decoder-only models.
    - model: VQAModel instance (has .model and .loss)
    - prompt_inputs: dict from model.encode(...)
    - labels_ids: LongTensor [B, T]
    - mask_prompt: when decoder-only with input_ids present, mask loss to answer tokens only
    """
    import torch  # local import to avoid surprises
    is_enc_dec = bool(getattr(getattr(model.model, "config", object()), "is_encoder_decoder", False))
    if is_enc_dec:
        _ = model.forward({**prompt_inputs, "labels": labels_ids})
        avg_nll = float(model.loss.item()) if getattr(model, "loss", None) is not None else float("inf")
        num_tokens = int(labels_ids.shape[0] * labels_ids.shape[1])
        return avg_nll, avg_nll * num_tokens, num_tokens

    input_ids = prompt_inputs.get("input_ids")
    attn = prompt_inputs.get("attention_mask")
    if input_ids is None or not mask_prompt:
        _ = model.forward({**prompt_inputs, "labels": labels_ids})
        avg_nll = float(model.loss.item()) if getattr(model, "loss", None) is not None else float("inf")
        num_tokens = int(labels_ids.shape[0] * labels_ids.shape[1])
        return avg_nll, avg_nll * num_tokens, num_tokens

    # Decoder-only: concatenate prompt + labels; mask prompt tokens
    full_ids = torch.cat([input_ids, labels_ids], dim=1)
    full_attn = torch.cat([attn, torch.ones_like(labels_ids)], dim=1) if attn is not None else None
    labels = torch.full_like(full_ids, -100)
    prompt_len = int(input_ids.shape[1])
    labels[:, prompt_len:] = full_ids[:, prompt_len:]

    model_inputs = dict(prompt_inputs)
    model_inputs["input_ids"] = full_ids
    if full_attn is not None:
        model_inputs["attention_mask"] = full_attn
    _ = model.forward({**model_inputs, "labels": labels})
    avg_nll = float(model.loss.item()) if getattr(model, "loss", None) is not None else float("inf")
    num_tokens = int(full_ids.shape[1] - prompt_len) * full_ids.shape[0]
    num_tokens = max(1, num_tokens)
    return avg_nll, avg_nll * num_tokens, num_tokens