"""Three-class metrics and AF operating points selected on validation only."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from metrics import compute_classification_metrics, print_classification_metrics
from engine import sequence_logits
from io_utils import write_json
from .data import CLASSES
from .clinical_metrics import binary_metrics, choose_operating_points, af_evaluation


def classification_metrics(y, pred, probability, num_classes=None):
    # train_event passes its original four-class argument. Ignore it here; the
    # logits and fixed metric labels are strictly SNR/AF/OTHER (three classes).
    result = compute_classification_metrics(y, pred, probability, 3)
    af = binary_metrics(np.asarray(y) == 1, np.asarray(pred) == 1, probability[:, 1])
    result["af_vs_non_af"] = {**af, "binary_f1": af["f1"], "decision_rule": "multiclass argmax is AF"}
    return result


@torch.no_grad()
def evaluate(loader, net, device, criterion, tag="Val"):
    net.eval()
    labels, probabilities, windows = [], [], []
    loss = 0.
    for data, target in loader:
        logits = sequence_logits(net, data)
        loss += float(criterion(logits, target.to(device)))
        labels.append(target.numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
        windows.extend([data.shape[1] // net.repeat_times] * len(target))
    y, probability = np.concatenate(labels), np.concatenate(probabilities)
    pred = probability.argmax(1)
    metrics = classification_metrics(y, pred, probability)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    rows = cm.sum(1, keepdims=True)
    report = {"records": len(y), "loss_mean_batches": loss / len(loader), "metrics": metrics,
              "class_names": CLASSES,
              "classification_report": classification_report(y, pred, labels=[0, 1, 2], target_names=CLASSES,
                                                              output_dict=True, zero_division=0),
              "confusion_matrix": cm.tolist(),
              "confusion_matrix_row_normalized": np.divide(cm, rows, out=np.zeros_like(cm, dtype=float),
                                                           where=rows != 0).tolist()}
    print_classification_metrics(tag, report["loss_mean_batches"], metrics)
    # Validation/test samplers are deterministic. Never obtain IDs from a
    # separately shuffled training iterator.
    order = [i for batch in loader.batch_sampler for i in batch]
    predictions = pd.DataFrame({"record_id": loader.dataset.labels.iloc[order].patient_id.values,
                                "label": y, "prediction": pred, "ecg_windows": windows})
    predictions["simulation_steps"] = predictions.ecg_windows * net.repeat_times
    for index, label in enumerate(CLASSES):
        predictions["p_" + label] = probability[:, index]
    return report, y, probability, predictions


def save_final_evaluation(out, val, test):
    # Best checkpoint is fixed before selecting any AF probability threshold.
    # Three-class argmax and validation macro-F1 selection remain unchanged.
    val_report, val_y, val_p, val_predictions = val
    write_json(val_report, out / "validation_metrics.json")
    val_predictions.to_csv(out / "validation_predictions.csv", index=False)
    report, y, probability, predictions = test
    points = choose_operating_points(val_y == 1, val_p[:, 1])
    write_json(points, out / "af_operating_points_validation.json")
    report["AF_vs_non_AF"] = af_evaluation(y, probability, points)
    write_json(report, out / "classification_metrics.json")
    predictions["af_decision_argmax"] = probability.argmax(1) == 1
    predictions["af_decision_threshold_0_5"] = probability[:, 1] >= .5
    for name, point in points.items():
        predictions["af_decision_" + name] = probability[:, 1] >= point["threshold"]
    for name in list(predictions.columns):
        if name.startswith("af_decision_"):
            predictions["af_missed_" + name.removeprefix("af_decision_")] = (y == 1) & ~predictions[name]
    predictions.to_csv(out / "test_predictions.csv", index=False)
    predictions[predictions.af_missed_argmax].to_csv(out / "af_missed_cases_argmax.csv", index=False)
    print("Test confusion matrix SNR/AF/OTHER:", report["confusion_matrix"], flush=True)
