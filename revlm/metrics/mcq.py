import numpy as np
import re
from ..dataset.utils import extract_choice_pairs, extract_choices


def _cm_and_acc(y_true_idx, y_pred_idx, num_classes: int = 4):
    """Return (acc, n, cm) given parallel lists of class indices (or None)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    total = 0
    correct = 0
    for gt, pr in zip(y_true_idx, y_pred_idx):
        if gt is None or pr is None:
            continue
        cm[gt, pr] += 1
        total += 1
        if gt == pr:
            correct += 1
    acc = (correct / total) if total > 0 else 0.0
    return acc, total, cm


def cm_metrics(cm: np.ndarray):
    """Compute per-class and macro precision, recall, F1, specificity from a confusion matrix.
    Returns a dict with per_class arrays and macro averages.
    """
    cm = np.asarray(cm)
    eps = 1e-12
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "specificity_per_class": specificity.tolist(),
        "f1_per_class": f1.tolist(),
        "macro_precision": float(np.nanmean(precision)),
        "macro_recall": float(np.nanmean(recall)),
        "macro_specificity": float(np.nanmean(specificity)),
        "macro_f1": float(np.nanmean(f1)),
    }


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

            # Choose predicted letter by explicit match first, else option-text
            if pred_letter in letters:
                y_pred_letters.append(letter_to_idx[pred_letter])
            elif pred_option and gold_label and pred_option.strip().lower() == str(gold_label).strip().lower():
                # If text matched gold, treat as correct letter
                y_pred_letters.append(letter_to_idx.get(gold_letter, None))
            else:
                y_pred_letters.append(None)
            y_true_letters.append(letter_to_idx.get(gold_letter, None))

    acc, total, cm = _cm_and_acc(y_true_letters, y_pred_letters, num_classes=4)
    return {"accuracy": acc, "n": total, "confusion_matrix": cm.tolist()}


def MCQ_metrics_score(model, vlmdataset):
    """Log-loss based accuracy using model.score_choices.

    Uses option texts from choices when available; otherwise falls back to ["A","B","C","D"].
    """
    loader = vlmdataset.loader
    letters = ["A", "B", "C", "D"]
    y_true_idx = []
    y_pred_idx = []

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

        # Score with model (per-example choice scores)
        results = model.score_choices(images, prompts, labels_per_ex, use_prob=True, use_avg=False)

        # results is List[Dict[label -> metrics]] aligned with labels_per_ex
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
                y_true_idx.append(gold_indices[i])
                y_pred_idx.append(best_idx)

    acc, total, cm = _cm_and_acc(y_true_idx, y_pred_idx, num_classes=4)
    return {"accuracy": acc, "n": total, "confusion_matrix": cm.tolist()}


def MCQ_metrics_classifier(model, vlmdataset):
    """Accuracy using single-step letter classification (fast path)."""
    loader = vlmdataset.loader
    letters = ["A", "B", "C", "D"]
    letter_to_idx = {c: i for i, c in enumerate(letters)}

    y_true_idx = []
    y_pred_idx = []

    for batch in loader:
        images = batch.get("images")
        prompts = batch.get("prompts")
        labels_text = batch.get("labels", [])
        choices_list = batch.get("choices", [])
        gold_letters_list = batch.get("label_letters", [])

        preds, _ = model.letter_classifier(images, prompts)

        for i, pred_letter in enumerate(preds):
            choices_str = choices_list[i] if i < len(choices_list) else ""
            gold_letter, _ = mcq_true_tuple(
                labels_text[i] if i < len(labels_text) else None,
                choices_str,
                gold_letters_list[i] if i < len(gold_letters_list) else None,
            )
            if not gold_letter or pred_letter not in letters:
                continue
            y_true_idx.append(letter_to_idx[gold_letter])
            y_pred_idx.append(letter_to_idx[pred_letter])

    acc, total, cm = _cm_and_acc(y_true_idx, y_pred_idx, num_classes=4)
    return {"accuracy": acc, "n": total, "confusion_matrix": cm.tolist()}

