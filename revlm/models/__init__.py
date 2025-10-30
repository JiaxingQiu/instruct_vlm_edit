import torch
import logging
from .wrapper import VQAModel

LOG = logging.getLogger(__name__)


def get_model(config):
    # Get model class name (with fallback to config model_name access)
    model_name = getattr(config.model, "name", "")
    model_class = getattr(config.model, "class_name", "VQAModel")
    model_pt = getattr(config.model, "pt", None)
    task = getattr(getattr(config, "experiment", None), "task", None) or getattr(config, "task", "vqa")
    if task == "vqa":
        LOG.info(f"Loading VQAModel for VQA task")
        model = VQAModel(config)
    else:
        raise NotImplementedError(f"Model class {model_class} for task {task} not implemented")
    
    if model_pt:
        LOG.info(f"Loading model from checkpoint {model_pt}")
        state_dict = torch.load(model_pt, map_location="cpu")
        model.model.load_state_dict(state_dict, strict=False)
    
    return model
