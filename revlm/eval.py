import argparse
import json
import os
import random
import torch
from PIL import Image

from .config_utils import *
from .dataset import *
from .models import *
from .metrics import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_eval(config, args):

    # Build model
    vlm = get_model(config)

    # Load dataset (test split)
    ds = get_dataset(config, split=args.split)
    if args.subsample and len(ds) > args.subsample:
        ds.data = random.sample(ds.data, args.subsample)
    
    # --- print 10 example answers ---
    choices = []
    imgs = []
    qs = []
    for i in range(10):
        ex = ds.data[i]
        print(ex)
        choices.append(ex["choices"])
        imgs.append(Image.open(ex["image"]).convert("RGB"))
        qs.append("Choose A/B/C/D based on the image. " + ex["question"] + ex['choices'])
    
    with torch.no_grad():
        ans = vlm.generate(images=imgs, prompts=qs, max_new_tokens=100)
    print(ans)
    # --------------

    # Prepare loader
    ds.set_dataloader(
        task=args.task,
        with_rationale=args.rationale,
        shuffle_choices=True,
        unpaired=True
    )

    if args.task == "mcq":
        # Run metrics
        res_text = MCQ_metrics_text(vlm, ds)
        res_score1 = MCQ_metrics_score(vlm, ds)
        res_score2 = MCQ_metrics_score(vlm, ds, score_by_letter=False)
        res_cls = MCQ_metrics_classifier(vlm, ds)
        results = {"text": res_text, "loss_letter": res_score1, "loss_option": res_score2, "classifier": res_cls}
    if args.task == "qa":
        # Run metrics
        res_text = QA_metrics_text(vlm, ds)
        res_loss = QA_metrics_loss(vlm, ds)
        res_qa = QA_metrics_nli_bi(vlm, ds)
        results = {"text": res_text, "loss": res_loss, "nli_bi": res_qa}

   

    # Save under res_dir
    res_dir = getattr(config, "res_dir")
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, f"{args.task}{'_rationale' if args.rationale else ''}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {out_path}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluation")
    parser.add_argument("--config", type=str, default="revlm/config/config.yaml", help="Path to YAML config file (CLI overrides YAML)")
    parser.add_argument("--editor", type=str, default="raw", choices=["raw", "ft", "ft_ewc", "ft_retrain", "mend", "grace", "rome", "memory", "defer"], help="Editor method to use ('raw' = no editing)")
    parser.add_argument("--model_name", type=str, default=None, help="Short VLM name to map to full HF id (e.g., 'qwen3', 'llava', 'blip')")
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name (overrides YAML if provided)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split to evaluate on")
    parser.add_argument("--task", type=str, default="mcq", choices=["mcq", "qa"], help="Task to evaluate")
    parser.add_argument("--rationale", action="store_true", help="Append rationale to prompts if available")
    parser.add_argument("--subsample", type=int, default=0, help="Evaluate on a random subset of this many examples (0=all)")
    
    args = parser.parse_args()
    config = configure_args(args, config_path=args.config)
    setattr(config, "device", device)
    run_eval(config, args)

