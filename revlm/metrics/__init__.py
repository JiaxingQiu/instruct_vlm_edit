import torch
import numpy as np
import re
from ..dataset.utils import extract_choice_pairs, extract_choices


def mcq_true_tuple(labels_text, choices_str, gold_letter=None):
    """Return (gold_letter, gold_label_text). gold_letter preferred if provided; otherwise derive from labels_text.
    choices_str is the MCQ string with (A)-(D).
    """
    pairs = extract_choice_pairs(choices_str or "")  # [(letter, option_text)]
    letter_to_text = {ltr: txt for ltr, txt in pairs}
    if gold_letter:
        gl = str(gold_letter).upper()
    else:
        gl = None
        gold_txt = str(labels_text or "")
        for ltr, txt in pairs:
            if gold_txt and txt.strip().lower() == gold_txt.strip().lower():
                gl = ltr
                break
    gold_label = (str(labels_text) if labels_text else None) or letter_to_text.get(gl)
    return gl, gold_label


def mcq_pred_tuple(output_text, choices_str):
    """Return (pred_letter_if_any, pred_option_if_any) from model output and choices.
    Detects explicit (A|B|C|D); also matches option text substring.
    """
    pairs = extract_choice_pairs(choices_str or "")
    pred_letter = None
    pred_option = None
    ot = str(output_text or "")
    m = re.search(r"\(([A-D])\)", ot, flags=re.IGNORECASE)
    if m:
        pred_letter = m.group(1).upper()
    ot_low = ot.lower()
    for ltr, txt in pairs:
        if txt and txt.lower() in ot_low:
            pred_option = txt
            if not pred_letter:
                pred_letter = ltr
            break
    return pred_letter, pred_option


def MCQ_metrics_text(model, vlmdataset):
    """Compute MCQ metrics using tuple logic.

    y_pred tuple: (pred_letter if any, pred_option if any)
    y_true tuple: (gold_letter, gold_label)
    Letter is taken from “(A|B|C|D)” in the output if present; otherwise, we match by option text substring.

    """
    loader = vlmdataset.loader
    letters = ["A", "B", "C", "D"]
    letter_to_idx = {c: i for i, c in enumerate(letters)}

    total = 0
    correct = 0
    y_true_letters = []
    y_pred_letters = []

    for batch in loader:
        images = batch.get("images")
        prompts = batch.get("prompts")
        labels_text = batch.get("labels", [])
        choices_list = batch.get("choices", [])
        gold_letters_list = batch.get("label_letters", [])

        outputs = model.generate(images, prompts, max_new_tokens=100)

        for i, out_text in enumerate(outputs):
            choices_str = choices_list[i] if i < len(choices_list) else ""
            gold_letter, gold_label = mcq_true_tuple(
                labels_text[i] if i < len(labels_text) else None,
                choices_str,
                gold_letters_list[i] if i < len(gold_letters_list) else None,
            )
            pred_letter, pred_option = mcq_pred_tuple(out_text, choices_str)

            # Accuracy: letter match OR option-text match
            if gold_letter:
                total += 1
                is_correct = False
                if pred_letter and pred_letter == gold_letter:
                    is_correct = True
                elif pred_option and gold_label and pred_option.strip().lower() == str(gold_label).strip().lower():
                    is_correct = True
                if is_correct:
                    correct += 1

                # For classification metrics on letters
                if pred_letter in letters:
                    y_pred_letters.append(letter_to_idx[pred_letter])
                else:
                    # mark as wrong by assigning an out-of-range pred? Simpler: skip confusion entry
                    y_pred_letters.append(None)
                if gold_letter in letters:
                    y_true_letters.append(letter_to_idx[gold_letter])
                else:
                    y_true_letters.append(None)

    # Build confusion matrix over A-D for valid pairs
    cm = np.zeros((4, 4), dtype=int)
    valid_pairs = 0
    for gt, pr in zip(y_true_letters, y_pred_letters):
        if gt is not None and pr is not None:
            cm[gt, pr] += 1
            valid_pairs += 1

    # Macro precision/recall/F1 over A-D
    eps = 1e-12
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = (tp / (tp + fp + eps))
        rec = (tp / (tp + fn + eps))
        f1 = (2 * prec * rec) / (prec + rec + eps)
    macro_precision = float(np.nanmean(prec))
    macro_recall = float(np.nanmean(rec))
    macro_f1 = float(np.nanmean(f1))

    acc = (correct / total) if total > 0 else 0.0
    return {
        "accuracy": acc,
        "n": total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }


def MCQ_metrics_score(model, vlmdataset):
    """Log-loss based accuracy using model.score_labels.

    Uses option texts from choices when available; otherwise falls back to ["A","B","C","D"].
    """
    loader = vlmdataset.loader
    letters = ["A", "B", "C", "D"]
    total = 0
    correct = 0

    for batch in loader:
        images = batch.get("images")
        prompts = batch.get("prompts")
        labels_text = batch.get("labels", [])
        choices_list = batch.get("choices", [])
        gold_letters = batch.get("label_letters", [])

        # Build label_words per example
        labels_per_ex = []
        gold_indices = []
        for i in range(len(prompts)):
            ch_str = choices_list[i] if i < len(choices_list) else ""
            ch = extract_choices(ch_str) if ch_str else []
            if len(ch) == 4:
                label_words = ch
                # gold index by letter or by matching label text
                if i < len(gold_letters) and gold_letters[i] in letters:
                    gold_idx = letters.index(str(gold_letters[i]).upper())
                else:
                    gold_txt = str(labels_text[i]) if i < len(labels_text) else ""
                    gold_idx = next((j for j, t in enumerate(ch) if t.strip().lower() == gold_txt.strip().lower()), None)
            else:
                label_words = letters
                if i < len(gold_letters) and gold_letters[i] in letters:
                    gold_idx = letters.index(str(gold_letters[i]).upper())
                else:
                    gold_idx = None
            labels_per_ex.append(label_words)
            gold_indices.append(gold_idx)

        # Score with model
        results = model.score_labels(images, prompts, labels_per_ex, use_prob=True, use_avg=False, temperature=1.0)

        # results is List[Dict[label -> metrics]]
        for i, per_lbl in enumerate(results):
            if not isinstance(per_lbl, dict) or gold_indices[i] is None:
                continue
            candidate_labels = labels_per_ex[i]
            # choose by prob if present else by lowest sum_nll
            best_idx = None
            best_val = None
            for j, w in enumerate(candidate_labels):
                met = per_lbl.get(w, {})
                val = met.get("prob")
                if val is None:
                    # use negative log-loss (lower is better)
                    val = -float(met.get("sum_nll", float("inf")))
                if best_val is None or val > best_val:
                    best_val = val
                    best_idx = j
            if best_idx is not None:
                total += 1
                if best_idx == gold_indices[i]:
                    correct += 1

    acc = (correct / total) if total > 0 else 0.0
    return {"accuracy": acc, "n": total}







def Accuracy(model, tokens):
    """Accuracy metric for classification tasks"""
    labels = tokens["labels"]
    new_tokens = {k: v for k, v in tokens.items() if k != "labels"}
    
    with torch.no_grad():
        outputs = model(**new_tokens)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.softmax(logits, -1).squeeze()
        argmaxs = torch.argmax(probs, dim=-1).squeeze()
        
        if labels.dim() == 0:
            return (labels == argmaxs).float()
        return (labels == argmaxs).float().mean()


def F1(model, batch):
    """F1 metric for VQA/generation tasks - token overlap"""
    input_ids = batch["input_ids"]
    preds = model.generate(input_ids, max_length=50).squeeze()
    
    if len(preds) > 1 and hasattr(model, "tokenizer"):
        preds = preds[preds != model.tokenizer.pad_token_id]
    
    labels = batch["labels"]
    gold_toks = labels[labels != -100].cpu().squeeze()
    
    if gold_toks.numel() == 0:
        return 0.0
    
    preds_np = preds.cpu().numpy() if torch.is_tensor(preds) else np.array(preds)
    gold_np = gold_toks.numpy() if torch.is_tensor(gold_toks) else np.array(gold_toks)
    
    num_same = len(np.intersect1d(preds_np, gold_np))
    
    if num_same == 0 or preds_np.size == 0:
        return 0.0
    if gold_np.size == 1 and preds_np.size == 1 and (gold_np == preds_np).all():
        return 1.0
    precision = num_same / preds_np.size
    recall = num_same / gold_np.size
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    return float(f1)


def is_acc_error(model, tokens):
    """Check if model prediction is incorrect (for classification)"""
    labels = tokens.get("labels")
    with torch.no_grad():
        new_tokens = {k: v for k, v in tokens.items() if k != "labels"}
        outputs = model(**new_tokens)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.softmax(logits, -1).squeeze()
        argmaxs = torch.argmax(probs, dim=-1).squeeze()
        return labels != argmaxs if labels.dim() == 0 else (labels != argmaxs).any()


def is_qa_error(model, tokens):
    """Check if model prediction is incorrect (for generation/VQA)"""
    f1_score = F1(model, tokens)
    return f1_score < 1.0


def get_metric(task):
    """Get metric function for given task"""
    if task == "vqa":
        return F1
    elif task == "captioning":
        return F1
    else:
        # Default to accuracy
        return Accuracy


def get_error_fn(task):
    """Get error checking function for given task"""
    if task in ["vqa", "captioning"]:
        return is_qa_error
    else:
        return is_acc_error

