from torch.utils.data import Dataset, DataLoader
import logging
import random
from PIL import Image

from .utils import *

LOG = logging.getLogger(__name__)


class VLMDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data = []
        self._load_data()
        # self.add_label_letter()
        

    def _load_data(self):
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_data")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def set_dataloader(self,
                        task="qa",  # "qa" or "mcq"
                        with_rationale=False,
                        shuffle_choices=False,
                        unpaired=False,
                        batch_size=32,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True):
        task = task.lower()
        if task == "mcq" and shuffle_choices:
            self.shuffle_choices(seed=333, unpaired=unpaired)
        
        for ex in self.data:
            if task == "mcq":
                sys_prompt = "Choose A/B/C/D based on the image."
                base = f"{sys_prompt} {ex['question']} {ex.get('choices','')}".strip()
            else:
                sys_prompt = "Answer the question based on the image."
                base = f"{sys_prompt} {ex['question']}".strip()
            ex["prompt"] = f"{base} {ex.get('rationale','')}".strip() if with_rationale else base

        # add label_letter: letter that matches the label in the choices column.  example: label = "car", choices = "(A) car\n(B) bike\n(C) train\n(D) bus" -> label_letter = "A"
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=self.image_collate)
        self.loader.task = task
        self.loader.with_rationale = with_rationale
        self.loader.shuffle_choices = shuffle_choices


    def shuffle_choices(self, seed=None, unpaired=False):
        """Shuffle (letter, option) pairs. If unpaired=False, also randomize letter–option association."""
        rng = random.Random(seed) if seed is not None else random
        for ex in self.data:
            chs = ex.get("choices", "")
            pairs = extract_choice_pairs(chs)
            if len(pairs) != 4:
                continue

            if not unpaired: # paired shuffle
                rng.shuffle(pairs)
            else: # unpaired shuffle
                letters = [ltr for ltr, _ in pairs]
                options = [opt for _, opt in pairs]
                rng.shuffle(letters)
                rng.shuffle(options)
                pairs = list(zip(letters, options))

            ex["choices"] = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])

    def add_label_letter(self):
        """Add 'label_letter' field per example by matching text in 'label' to current choices.
        Expects 'label' to contain the answer text (e.g., 'cab').
        """
        for ex in self.data:
            ans_text = str(ex.get("label", "")).strip()
            pairs = extract_choice_pairs(ex.get("choices", ""))
            letter = None
            for ltr, opt in pairs:
                if opt.strip().lower() == ans_text.lower():
                    letter = ltr
                    break
            ex["label_letter"] = letter

    def image_collate(self, batch):
        """Collate function that loads images and returns a batch dict.
        Expects items with keys: 'image' (path), 'prompt' (string), optional 'label'.
        """
        images = [Image.open(ex["image"]).convert("RGB") for ex in batch]
        prompts = [ex.get("prompt", ex.get("question", "")) for ex in batch]
        label = [ex.get("label") for ex in batch]
        choices = [ex.get("choices", "") for ex in batch]
        label_letter = [ex.get("label_letter") for ex in batch]
        return {
            "images": images,
            "prompts": prompts,
            "label": label,
            "choices": choices,
            "label_letter": label_letter,
        }


class AOKVQADataset(VLMDataset):
    def __init__(self, split: str = "train"):
        super().__init__(split=split)

    def _load_data(self):
        split = self.split if self.split in ("train", "val", "test") else "train"
        split_paths = data_download_parquet_splits(
            repo_id="JJoy333/RationaleVQA",
            path_in_repo="AOKVQA",
        )
        df = data_load_split_df(split_paths.get(split))
        self.data = data_rows_to_examples(df)
        if not self.data:
            LOG.warning("AOKVQADataset split '%s' is empty.", split)


class FVQADataset(VLMDataset):
    def __init__(self, split: str = "train"):
        super().__init__(split=split)

    def _load_data(self):
        split = self.split if self.split in ("train", "val", "test") else "train"
        split_paths = data_download_parquet_splits(
            repo_id="JJoy333/RationaleVQA",
            path_in_repo="FVQA",
        )
        df = data_load_split_df(split_paths.get(split))
        self.data = data_rows_to_examples(df)
        if not self.data:
            LOG.warning("FVQADataset split '%s' is empty.", split)


def get_dataset(config, split="train"):
    dataset_name = getattr(getattr(config, "experiment", {}), "dataset_name", "").lower()

    # Prefer explicit dataset_name if provided
    if dataset_name == "aokvqa":
        edit_dataset = AOKVQADataset(split=split)
    elif dataset_name == "fvqa":
        edit_dataset = FVQADataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return edit_dataset
