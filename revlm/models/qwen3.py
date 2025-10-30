import torch
from PIL import Image
import transformers
from .utils import ckpt_dir


class Qwen3VLM(torch.nn.Module):
    """Wrapper for Qwen3-VL models with chat-template generation."""
    def __init__(self, config):
        super().__init__()
        # Load processor and model with safe defaults
        self.processor = transformers.AutoProcessor.from_pretrained(
            config.model.name,
            cache_dir=ckpt_dir(),
            trust_remote_code=True,
        )
        ModelClass = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if ModelClass is None:
            # Fallback to a generic auto class if specific not present
            ModelClass = getattr(transformers, "AutoModelForVision2Seq", transformers.AutoModel)
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }
        try:
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else None
        except Exception:
            pass
        model_path = getattr(config.model, "pt", None) or config.model.name
        self.model = ModelClass.from_pretrained(
            model_path,
            **{k: v for k, v in load_kwargs.items() if v is not None},
        ).eval()

        self.tokenizer = getattr(self.processor, "tokenizer", None)
        device_str = getattr(config, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str) if isinstance(device_str, str) else device_str

    def forward(self, **inputs):
        return self.model(**inputs)

    def generate(self, images, prompts, max_new_tokens=50, **kwargs):
        # Normalize inputs
        images = [images] if isinstance(images, Image.Image) else images
        prompts = [prompts] if isinstance(prompts, str) else prompts

        if not hasattr(self.processor, "apply_chat_template"):
            # Fall back to default pathway if needed
            proc_inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True)
            proc_inputs = {k: v.to(self.device) for k, v in proc_inputs.items()}
        else:
            msgs = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": prompts[0]},
                ],
            }]
            proc_inputs = self.processor.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**proc_inputs, max_new_tokens=max_new_tokens, **kwargs)
            return self.processor.batch_decode(gen_ids, skip_special_tokens=True)


