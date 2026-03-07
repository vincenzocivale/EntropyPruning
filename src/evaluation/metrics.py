import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_tar_at_far(scores, is_correct, far_threshold=1e-4):
    """TAR@FAR for multiclass classification.

    Args:
        scores: max softmax probability per sample
        is_correct: boolean array, True if prediction is correct
        far_threshold: target false acceptance rate

    Returns:
        (tar, threshold)
    """
    scores = np.array(scores)
    correct = np.array(is_correct).astype(bool)
    incorrect = ~correct

    if incorrect.sum() == 0:
        return 1.0, float('nan')

    n_incorrect = incorrect.sum()
    n_far_allowed = max(1, int(np.ceil(n_incorrect * far_threshold)))
    sorted_wrong = np.sort(scores[incorrect])[::-1]
    threshold = sorted_wrong[min(n_far_allowed - 1, len(sorted_wrong) - 1)]

    tar = (scores[correct] >= threshold).mean()
    return float(tar), float(threshold)


def evaluate(model, loader, device, far_threshold=1e-4):
    """Evaluate model: accuracy, F1 macro, TAR@FAR.

    Returns:
        dict with keys: acc, f1_macro, tar_at_far, threshold
    """
    model.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs = logits.softmax(-1)
            preds = probs.argmax(-1).cpu()
            score = probs.max(-1).values.cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_scores.append(score)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_scores = torch.cat(all_scores).numpy()

    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    is_correct = (all_preds == all_labels)
    tar, thr = compute_tar_at_far(all_scores, is_correct, far_threshold)

    return {"acc": acc, "f1_macro": f1, "tar_at_far": tar, "threshold": thr}
