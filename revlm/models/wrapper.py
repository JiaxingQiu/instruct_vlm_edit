import torch
import logging
from PIL import Image
from .utils import *

LOG = logging.getLogger(__name__)


class VQAModel(torch.nn.Module):
    """Vision Question Answering model wrapper - works with all VLMs"""
    def __init__(self, config):
        super(VQAModel, self).__init__()
        self.config = config

        self.model = get_hf_model(config)
        self.processor = get_processor(config)
        self.tokenizer = get_tokenizer(config)
        self.preprocess = get_preprocess(config)
        self.device = config.device
        self.model.eval()

    def forward(self, **inputs):
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return self.model(**inputs)

    def generate(self, images, prompts, max_new_tokens=50, **kwargs):
        images = [images] if isinstance(images, Image.Image) else images
        prompts = [prompts] if isinstance(prompts, str) else prompts
        
        # preprocess images and prompts into tensors (CPU), then move to device
        inputs = self.preprocess(images, prompts, self.processor)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # generate text
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
            outputs = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        return outputs