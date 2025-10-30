import torch
import logging
from PIL import Image
from .utils import build_vlm

LOG = logging.getLogger(__name__)


class VQAModel(torch.nn.Module):
    """Vision Question Answering model wrapper - works with all VLMs"""
    def __init__(self, config):
        super(VQAModel, self).__init__()
        self.config = config
        self.model, self.processor, self.tokenizer, self.inner = build_vlm(config)
        
        # Device handling
        device_str = getattr(config, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str) if isinstance(device_str, str) else device_str
        
        # Model is already placed via device_map in from_pretrained; just eval
        self.model.eval()

    def forward(self, **inputs):
        """Forward pass for VQA"""
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return self.model(**inputs)

    def generate(self, images, prompts, max_new_tokens=50, **kwargs):
        """Generate answers for vision question answering"""
        # Normalize inputs
        images = [images] if isinstance(images, Image.Image) else images
        prompts = [prompts] if isinstance(prompts, str) else prompts
        
        # Delegate to inner wrapper if present
        if self.inner is not None:
            return self.inner.generate(images, prompts, max_new_tokens=max_new_tokens, **kwargs)

        # Generic processing path
        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            generated_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text