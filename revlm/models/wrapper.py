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
        self.device = config.device

        self.model = get_hf_model(config)
        self.model.eval()
        self.processor = get_processor(config)
        self.tokenizer = get_tokenizer(config)
        self.preprocess = get_preprocess(config)
        
    
    def forward(self, **inputs):
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return self.model(**inputs)

    
    def encode(self, images, prompts, tokenize=False):
        images = [images] if isinstance(images, Image.Image) else images
        prompts = [prompts] if isinstance(prompts, str) else prompts
        
        # preprocess images and prompts into tensors (CPU), then move to device
        inputs = self.preprocess(images, prompts, self.processor, tokenize=tokenize)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs
        

    def generate(self, images, prompts, tokenize=False, **kwargs):
        inputs = self.encode(images, prompts, tokenize=tokenize)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
            outputs_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            answers = [clean_answer(o, i) for (o, i) in zip(outputs_text, prompts)]
        return answers