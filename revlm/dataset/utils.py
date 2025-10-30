import os
import logging
from typing import List, Dict, Optional, Tuple
import re

import pandas as pd
from huggingface_hub import snapshot_download

LOG = logging.getLogger(__name__)


def data_download_parquet_splits(repo_id: str, path_in_repo: str, cache_dir: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Download train/val/test parquet files from a HF dataset directory."""
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{path_in_repo}/*.parquet"],
        cache_dir=cache_dir,
    )
    base_dir = os.path.join(local_root, path_in_repo)
    return {
        split: os.path.join(base_dir, f"{split}.parquet") if os.path.exists(os.path.join(base_dir, f"{split}.parquet")) else None
        for split in ("train", "val", "test")
    }


def data_load_split_df(parquet_path: Optional[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(columns=["image_path", "question", "answer", "rationale", "choices"]) if parquet_path is None
        else pd.read_parquet(parquet_path)
    )


def data_rows_to_examples(df: pd.DataFrame) -> List[Dict]:
    """Convert a dataframe to trainer-ready dicts with optional fields preserved."""
    required = ["image_path", "question", "answer", "rationale", "choices"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")
    if df.empty:
        return []

    records = df[required].to_dict(orient="records")
    return [
        {
            "image": r["image_path"],
            "question": r["question"],
            "label": r["answer"],
            "rationale": r["rationale"],
            "choices": r["choices"],
        }
        for r in records
    ]


# Tokenization utilities moved from top-level utils
def tokenize_vlm(batch, tokenizer, device, test=False):
    """
    Tokenize VLM input batch.
    Assumes batch contains:
    - image: image path or tensor
    - question: question prompt/question
    - label: target answer (optional)
    """
    question = batch.get("question", batch["question"]) if isinstance(batch, dict) else ""
    label = batch.get("label", None) if isinstance(batch, dict) else None

    tokens = tokenizer(
        question if isinstance(question, list) else [question],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    if label is not None and not test:
        label_tokens = tokenizer(
            label if isinstance(label, list) else [label],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens["label"] = label_tokens["input_ids"]
        tokens["label"][tokens["label"] == tokenizer.pad_token_id] = -100
    elif not test:
        tokens["label"] = tokens["input_ids"].clone()
        if tokenizer.pad_token_id is not None:
            tokens["label"][tokens["input_ids"] == tokenizer.pad_token_id] = -100

    tokens = {k: v.to(device) for k, v in tokens.items()}
    return tokens


def get_tokenize_fn(task):
    """Get tokenization function for given task"""
    return tokenize_vlm



def extract_choice_pairs(s: str):
    """Order-agnostic parse of lines like '(A) foo', '(B) bar', ...
    Returns list of (letter, text) in the order they appear.
    """
    pairs = re.findall(r"\(([A-D])\)\s*(.+)", s)
    return [(ltr, txt.strip()) for (ltr, txt) in pairs]


def extract_choices(question: str):
    """Order-agnostic: return only the option texts in the order they appear."""
    return [txt for (_ltr, txt) in extract_choice_pairs(question)]


