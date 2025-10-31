import torch
import logging
import math
from PIL import Image
from .utils import *
from typing import Dict

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
        
    
    def forward(self, batch):
        """Accept a batch from dataset.loader and return logits.
        If batch has 'images' and 'prompts', encode internally; else assume tokenized.
        Stores loss if provided by the HF model.
        """
        if isinstance(batch, dict) and ("images" in batch and "prompts" in batch):
            inputs = self.encode(batch["images"], batch["prompts"], tokenize=False)
        else:
            inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        output = self.model(**inputs)
        self.loss = getattr(output, "loss", None)
        return output.logits if hasattr(output, "logits") else output

    
    def encode(self, images, prompts, tokenize=False):
        images = [images] if isinstance(images, Image.Image) else images
        prompts = [prompts] if isinstance(prompts, str) else prompts
        
        # preprocess images and prompts into tensors (CPU), then move to device
        inputs = self.preprocess(images, prompts, self.processor, tokenize=tokenize)
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs
        

    def generate(self, images, prompts, **kwargs):
        inputs = self.encode(images, prompts, tokenize=False)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
            outputs_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
            answers = [clean_answer(o, i) for (o, i) in zip(outputs_text, prompts)]
        return answers

    
    def _nll_to_probs(self, label_losses: Dict[str, Dict[str, float]], use_avg: bool = False, temperature: float = 1.0) -> Dict[str, float]:
        scores = {}
        temp = max(1e-8, float(temperature))
        for lbl, d in label_losses.items():
            nll = float(d['avg_nll'] if use_avg else d['sum_nll'])
            scores[lbl] = -(nll) / temp
        m = max(scores.values()) if scores else 0.0
        exps = {lbl: math.exp(s - m) for lbl, s in scores.items()}
        Z = sum(exps.values()) or 1.0
        return {lbl: v / Z for lbl, v in exps.items()}

    
    @torch.no_grad()
    def _score_label_words(self,
                            image,
                            prompt,
                            label_words,
                            use_prob: bool = True,
                            use_avg: bool = False,
                            temperature: float = 1.0,
                        ):
        # Build model-ready inputs using existing encode (keep tokenize=False to return strings to processor)
        prompt_inputs = self.encode(image, prompt, tokenize=False)

        is_enc_dec = bool(getattr(getattr(self.model, "config", object()), "is_encoder_decoder", False))
        results = {}

        for word in label_words:
            cand_ids = self.tokenizer(
                word,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)

            if is_enc_dec:
                # Use forward to run model and populate self.loss
                self.forward({**prompt_inputs, "labels": cand_ids})
                avg_nll = float(self.loss.item()) if self.loss is not None else float("inf")
                num_tokens = int(cand_ids.shape[1])
                sum_nll = avg_nll * num_tokens
            else:
                input_ids = prompt_inputs.get("input_ids")
                attn = prompt_inputs.get("attention_mask")

                if input_ids is None:
                    self.forward({**prompt_inputs, "labels": cand_ids})
                    avg_nll = float(self.loss.item()) if self.loss is not None else float("inf")
                    num_tokens = int(cand_ids.shape[1])
                    sum_nll = avg_nll * num_tokens
                else:
                    full_ids = torch.cat([input_ids, cand_ids], dim=1)
                    full_attn = torch.cat([attn, torch.ones_like(cand_ids)], dim=1) if attn is not None else None

                    labels = torch.full_like(full_ids, -100)
                    prompt_len = int(input_ids.shape[1])
                    labels[:, prompt_len:] = full_ids[:, prompt_len:]

                    model_inputs = dict(prompt_inputs)
                    model_inputs["input_ids"] = full_ids
                    if full_attn is not None:
                        model_inputs["attention_mask"] = full_attn

                    self.forward({**model_inputs, "labels": labels})
                    avg_nll = float(self.loss.item()) if self.loss is not None else float("inf")
                    num_tokens = int(full_ids.shape[1] - prompt_len)
                    sum_nll = avg_nll * max(1, num_tokens)

            results[word] = {
                "sum_nll": float(sum_nll),
                "avg_nll": float(avg_nll),
                "num_tokens": float(num_tokens),
            }

        if use_prob and results:
            probs = self._nll_to_probs(results, use_avg=use_avg, temperature=temperature)
            for lbl, p in probs.items():
                if lbl in results:
                    results[lbl]["prob"] = float(p)

        return results

    
    @torch.no_grad()
    def score_labels(
                    self,
                    images,
                    prompts,
                    label_words,
                    use_prob: bool = True,
                    use_avg: bool = False,
                    temperature: float = 1.0,
                ):
        # Batched mode only: lists for images/prompts and a list of label lists
        assert isinstance(images, list) and isinstance(prompts, list) and isinstance(label_words, list), "Expect lists for images, prompts, and label_words"
        assert len(images) == len(prompts) == len(label_words), "images/prompts/label_words must have same length"
        return [
            self._score_label_words(img, pr, lbls, use_prob=use_prob, use_avg=use_avg, temperature=temperature)
            for img, pr, lbls in zip(images, prompts, label_words)
        ]


