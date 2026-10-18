"""Summaries from completed CPSC experiments; missing energy is not zero."""
import argparse
from pathlib import Path
import pandas as pd
from io_utils import read_json


def summarize(root):
    root = Path(root)
    rows, af_rows = [], []
    for marker in sorted(root.glob("seed*/*/complete.json")):
        out = marker.parent
        meta = read_json(out / "run_config.json")
        c = meta["config"]
        if c.get("dataset") != "cpsc2018" or not str(c.get("data_protocol", "")).startswith("af_priority_"):
            continue
        report = read_json(out / "classification_metrics.json")
        history = read_json(out / "training_history.json")
        efficiency = read_json(out / "efficiency_metrics.json") if (out / "efficiency_metrics.json").exists() else {}
        grouped = efficiency.get("grouped_inference", {})
        base = {"dataset": "cpsc2018", "duration": c["duration"], "variant": c["variant"],
                "train_seed": c["train_seed"], "split_seed": c["split_seed"], "leads": c["leads"],
                "repeat_times": c["repeat_times"], "window_length": c["window_length"],
                "preprocessing": c["preprocessing"], "tail_mode": c["tail_mode"], "split_mode": c["split_mode"],
                "data_protocol": c["data_protocol"], "weight_decay": c["weight_decay"]}
        metrics = report["metrics"]
        row = {**base, "epochs": len(history), "best_epoch": read_json(marker)["best_epoch"]}
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "auc_ovr_macro", "auc_ovr_weighted"):
            row[key] = metrics[key]
        row["best_val_macro_f1"] = read_json(marker)["best_val_macro_f1"]
        af = metrics["af_vs_non_af"]
        for key in ("sensitivity", "specificity", "auc", "f1", "missed_af_count", "missed_af_fraction"):
            row["af_" + key] = af[key]
        potential = sum(x["potential_optimizer_steps"] for x in history)
        updates = sum(x["optimizer_steps"] for x in history)
        row.update(potential_optimizer_steps=potential, optimizer_steps=updates, update_skip_fraction=1 - updates / potential,
                   training_wall_s=sum(x["epoch_wall_time_s"] for x in history),
                   historical_trace_cache_peak_bytes=max(x["historical_trace_cache_peak_bytes"] for x in history),
                   full_resolution_trace_equivalent_peak_bytes=max(x["full_resolution_trace_equivalent_peak_bytes"] for x in history),
                   training_peak_cuda_memory_mb=max((x["peak_cuda_memory_allocated_mb"] for x in history
                                                     if x["peak_cuda_memory_allocated_mb"] is not None), default=None),
                   latency_ms_per_record=grouped.get("latency_ms_per_record"))
        energies = [x.get("gpu_energy_j") for x in history]
        row["training_gpu_energy_j"] = sum(energies) if all(x is not None for x in energies) else None
        row["training_seconds_per_epoch"] = row["training_wall_s"] / len(history)
        row["optimizer_steps_per_epoch"] = updates / len(history)
        row["inference_peak_allocated_mib"] = grouped.get("peak_cuda_memory_allocated_mb")
        row["dense_gmac_per_record"] = efficiency.get("operations", {}).get("conv_linear_macs_per_record_mean")
        if row["dense_gmac_per_record"] is not None: row["dense_gmac_per_record"] /= 1e9
        row["firing_fraction"] = efficiency.get("overall_lif_firing_fraction")
        rows.append(row)
        evaluation = report["AF_vs_non_AF"]
        points = {"argmax": evaluation["argmax"], "threshold_0_5": evaluation["threshold_0_5"]}
        points.update({name: point["test"] for name, point in evaluation["operating_points"].items()})
        for name, point in points.items():
            af_rows.append({**base, "operating_point": name,
                            **{key: point[key] for key in ("sensitivity", "specificity", "f1", "auc", "missed_af_count", "missed_af_fraction")}})
    if not rows:
        return
    groups = ["duration", "variant", "leads", "repeat_times", "window_length", "preprocessing", "tail_mode", "split_mode", "data_protocol", "weight_decay"]
    _save(rows, root / "summary.csv", root / "summary_mean_std.csv", groups)
    _save(af_rows, root / "af_operating_points.csv", root / "af_operating_points_mean_std.csv", groups + ["operating_point"])


def _save(rows, details, summary, keys):
    frame = pd.DataFrame(rows)
    frame.to_csv(details, index=False)
    metadata = {"dataset", "duration", "variant", "train_seed", "split_seed", "leads",
                "repeat_times", "window_length", "preprocessing", "tail_mode", "split_mode", "operating_point", "data_protocol", "weight_decay"}
    numeric = [key for key in frame.columns if key not in metadata]
    # Entirely missing GPU measurements otherwise have object dtype and vanish
    # from pandas' numeric selection. Keep them explicitly with count=0.
    for key in numeric:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    summaries = []
    for group, values in frame.groupby(keys, sort=True):
        row = dict(zip(keys, group))
        row.update(n_seeds=len(values), train_seeds=",".join(map(str, values.train_seed)))
        row["split_seeds"] = ",".join(map(str, values.split_seed))
        row["seed_pairs"] = ",".join(f"{t}:{s}" for t, s in zip(values.train_seed, values.split_seed))
        for key in numeric:
            column = values[key].dropna()
            row[key + "_n"] = len(column)
            row[key + "_mean"] = float(column.mean()) if len(column) else None
            row[key + "_std"] = float(column.std(ddof=1)) if len(column) > 1 else None
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(summary, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    summarize(p.parse_args().root)
