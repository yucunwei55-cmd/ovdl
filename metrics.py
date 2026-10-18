"""固定 N/A/O/~ 顺序；宏/加权指标和 AF 二分类指标，不做时长分组。"""
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             roc_auc_score,classification_report,confusion_matrix)

def compute_classification_metrics(y_true, y_pred, y_proba, num_classes=None):
    """Fixed class order; unavailable AUC is null rather than an invented zero."""
    if num_classes is None:
        num_classes = y_proba.shape[1] if y_proba.ndim == 2 else 0
    class_ids = list(range(num_classes))
    metrics = {'accuracy': float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0}
    for average in ('macro', 'weighted'):
        for name, fn in [('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]:
            metrics[f'{name}_{average}'] = float(fn(y_true, y_pred, labels=class_ids,
                average=average, zero_division=0)) if len(y_true) else 0.0
        auc = None
        if len(y_true) and num_classes >= 2 and len(np.unique(y_true)) == num_classes:
            try:
                if num_classes == 2:
                    value = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    value = roc_auc_score(y_true, y_proba, labels=class_ids, multi_class='ovr', average=average)
                if np.isfinite(value):
                    auc = float(value)
            except ValueError:
                pass
        metrics[f'auc_ovr_{average}'] = auc
    if num_classes == 4 and len(y_true):
        metrics['f1_NAO'] = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro', zero_division=0))
        target = np.asarray(y_true) == 1
        decision = np.asarray(y_pred) == 1
        tp = int(np.sum(target & decision)); fn = int(np.sum(target & ~decision))
        tn = int(np.sum(~target & ~decision)); fp = int(np.sum(~target & decision))
        metrics['af_vs_non_af'] = {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None,
            'binary_f1': float(f1_score(target, decision, zero_division=0)),
            'auc': float(roc_auc_score(target, y_proba[:, 1])) if len(np.unique(target)) == 2 else None,
            'decision_rule': 'multiclass argmax is AF',
            'noise_included_in_non_af': True}
    return metrics

def print_classification_metrics(tag, loss, metrics):
    print(f'{tag} Loss(mean batches): {loss:.4f} | Accuracy: {metrics["accuracy"]:.4f}')
    for average in ('macro', 'weighted'):
        auc = metrics[f'auc_ovr_{average}']
        auc_text = f'{auc:.4f}' if auc is not None else 'unavailable (missing classes / invalid probabilities)'
        print(f'  {average}: Precision={metrics[f"precision_{average}"]:.4f} | '
              f'Recall={metrics[f"recall_{average}"]:.4f} | F1={metrics[f"f1_{average}"]:.4f} | '
              f'AUC(OVR)={auc_text}')
