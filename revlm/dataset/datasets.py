import os
from torch.utils.data import Dataset
import logging
import re
import random

from .utils import data_download_parquet_splits, data_load_split_df, data_rows_to_examples

LOG = logging.getLogger(__name__)


class VLMDataset(Dataset):
    def __init__(self, split="train"):
        self.split = split
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """Load dataset - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _load_data")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    
    def shuffle_choices(self, seed=None):
        """Shuffle (letter, option) pairs together, preserving their association.
        The rendered lines may start with any of (A|B|C|D) after shuffling.
        """
        rng = random.Random(seed) if seed is not None else random
        for ex in self.data:
            chs = ex.get("choices", "")
            pairs = extract_choice_pairs(chs)
            if len(pairs) != 4:
                continue
            rng.shuffle(pairs)
            ex["choices"] = "\n".join([f"({ltr}) {txt}" for (ltr, txt) in pairs])

    def add_letter_labels(self):
        """Add 'letter_label' field per example by matching text in 'labels' to current choices.
        Expects 'labels' to contain the answer text (e.g., 'cab').
        """
        for ex in self.data:
            ans_text = str(ex.get("labels", "")).strip()
            pairs = extract_choice_pairs(ex.get("choices", ""))
            letter = None
            for ltr, opt in pairs:
                if opt.strip().lower() == ans_text.lower():
                    letter = ltr
                    break
            ex["letter_label"] = letter


def extract_choice_pairs(s: str):
    """Order-agnostic parse of lines like '(A) foo', '(B) bar', ...
    Returns list of (letter, text) in the order they appear.
    """
    pairs = re.findall(r"\(([A-D])\)\s*(.+)", s)
    return [(ltr, txt.strip()) for (ltr, txt) in pairs]


def extract_choices(question: str):
    """Order-agnostic: return only the option texts in the order they appear."""
    return [txt for (_ltr, txt) in extract_choice_pairs(question)]


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
