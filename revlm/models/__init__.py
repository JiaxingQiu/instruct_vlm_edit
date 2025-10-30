import torch
import logging
from .wrapper import VQAModel

LOG = logging.getLogger(__name__)


def get_model(config):
    """
    Factory function to get model based on config.
    Args:
        config: NestedConfig with model settings (config.model.name, config.model.class_name, etc.)
    Returns:
        Model instance (VQAModel for VQA tasks)
    """
    # Get model class name (with fallback to config model_name access)
    if hasattr(config, "model"):
        model_name = getattr(config.model, "name", "")
        model_class = getattr(config.model, "class_name", "VQAModel")
        model_pt = getattr(config.model, "pt", None)
    else:
        # Fallback for backward compat
        model_name = getattr(config, "model_name", "")
        model_class = getattr(config, "model_class", "VQAModel")
        model_pt = getattr(config, "model_pt", None)
    
    # For VQA task, use VQAModel wrapper
    task = getattr(getattr(config, "experiment", None), "task", None) or getattr(config, "task", "vqa")
    
    if task == "vqa" or model_class == "VQAModel" or model_class in [None, ""]:
        LOG.info(f"Loading VQAModel for VQA task")
        model = VQAModel(config)
    else:
        raise NotImplementedError(f"Model class {model_class} for task {task} not implemented")
    
    # Load from checkpoint if provided
    if model_pt:
        LOG.info(f"Loading model from checkpoint {model_pt}")
        try:
            state_dict = torch.load(model_pt, map_location="cpu")
            model.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            LOG.warning(f"Could not load checkpoint from {model_pt}: {e}")
    
    return model
