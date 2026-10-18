"""CPSC2018 training using the unchanged V2 model, traces and event engine."""
import fcntl
import platform
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sampling import seed_everything, seed_worker
from model import from_config
from engine import train_event
from event_controller import EventController
from metrics import print_classification_metrics
from io_utils import read_json, write_json, save_checkpoint, load_checkpoint, sha256
from .config import parse_args, validate_config
from .data import ECGDataset, TimeStepGroupSampler, collate, CLASSES
from .protocol import prepare_splits
from .evaluation import classification_metrics, evaluate, save_final_evaluation
from .resources import profile


def build_loaders(a):
    datasets, loaders = {}, {}
    generator = torch.Generator().manual_seed(a.train_seed)
    for name in ("train", "val", "test"):
        ds = ECGDataset(name, a.data_dir, Path(a.out_dir) / "splits" / (name + ".csv"),
                        a.duration, a.window_length, a.repeat_times, a.train_seed, not a.no_augment, a.leads, a.preprocessing, a.tail_mode)
        sampler = TimeStepGroupSampler(ds.time_steps_list, a.batch_size, a.train_seed, name == "train")
        datasets[name] = ds
        loaders[name] = DataLoader(ds, batch_sampler=sampler, num_workers=a.num_workers, pin_memory=True,
                                  collate_fn=collate, worker_init_fn=seed_worker, generator=generator,
                                  **({"multiprocessing_context": "spawn"} if a.num_workers else {}))
    return datasets, loaders, generator


def run(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(a, out)


def execute(a, out):
    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)
    seed_everything(a.train_seed)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
    config = vars(a).copy()
    metadata = out / "run_config.json"
    if metadata.exists():
        validate_config(read_json(metadata)["config"], config)
    hashes, manifest = prepare_splits(a, out)
    history_path = out / "training_history.json"
    history = read_json(history_path) if history_path.exists() else []
    completed_epoch = history[-1]["epoch"] if history else 0
    if completed_epoch > a.epochs:
        raise ValueError("Requested epochs are fewer than already completed epochs")
    marker = out / "complete.json"
    required = [out / name for name in ("classification_metrics.json", "af_operating_points_validation.json", "test_predictions.csv", "validation_metrics.json", "validation_predictions.csv")]
    if not a.no_profile_efficiency:
        required.append(out / "efficiency_metrics.json")
    if completed_epoch == a.epochs and marker.exists() and all(p.exists() for p in required) and (out / "best.pth").exists():
        status = read_json(marker)
        if status["epochs"] == a.epochs and status["best_sha256"] == sha256(out / "best.pth"):
            print("SKIP completed:", out, flush=True)
            return
    write_json({"config": config, "split_sha256": hashes, "reference_sha256": manifest["label_csv_sha256"],
                "diagnosis_reference_sha256": manifest["reference_csv_sha256"],
                "class_mapping": dict(zip(CLASSES, range(3))), "patient_split": False, "split_unit": "record",
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                                "cuda": torch.version.cuda, "device": str(device),
                                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}, metadata)
    print("CPSC label audit:", manifest["label_audit"], flush=True)
    datasets, loaders, generator = build_loaders(a)
    write_json({name: {"records": len(ds), "ecg_window_count_histogram": dict(sorted(Counter(ds.window_counts).items())),
                       "total_simulation_steps": sum(ds.window_counts) * a.repeat_times,
                       "batches": len(loaders[name])} for name, ds in datasets.items()}, out / "duration_stats.json")
    net = from_config(config, datasets["train"].nleads, device)
    net.event_controller = EventController(a)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=.5)
    first, best_epoch, best_val = 1, -1, -1.
    if (out / "last.pth").exists():
        ck = load_checkpoint(out / "last.pth")
        validate_config(ck["config"], config)
        if ck["split_sha256"] != hashes or ck["epoch"] > a.epochs:
            raise ValueError("Checkpoint split/epoch mismatch")
        net.load_state_dict(ck["model_state_dict"])
        net.event_controller.load_state_dict(ck["event_controller_state"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        history, first = ck["training_history"], ck["epoch"] + 1
        best_epoch, best_val = ck["best_epoch"], ck["best_val_f1"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        generator.set_state(ck["loader_rng"].cpu())
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"RESUME at epoch {first}", flush=True)
        del ck
    elif history or (out / "best.pth").exists():
        raise ValueError("Existing results without last.pth; refusing to restart over them")
    if first <= a.epochs and marker.exists():
        marker.unlink()
    trained = False
    for epoch in range(first, a.epochs + 1):
        trained = True
        # Imported CPSC augmentation and sampler use (seed, epoch), independently
        # of model RNG. Set before spawning DataLoader workers each epoch.
        datasets["train"].epoch = epoch
        loaders["train"].batch_sampler.epoch = epoch
        row = train_event(loaders["train"], net, criterion, epoch, scheduler, optimizer, device, a,
                          metrics_fn=classification_metrics, print_metrics=print_classification_metrics)
        for key in ("per_class_sample_step_opportunities", "per_class_sample_steps_with_batch_update", "per_class_sample_events"):
            row[key] = row[key][:3]  # Original shared engine's unused fourth class is always zero.
        row["validation"] = evaluate(loaders["val"], net, device, criterion)[0]
        row["validation"]["selective_state"] = net.boundary_report()
        history.append(row)
        f1 = row["validation"]["metrics"]["f1_macro"]
        improved = f1 > best_val
        if improved:
            best_epoch, best_val = epoch, f1
        ck = {"model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
              "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best_epoch": best_epoch,
              "best_val_f1": best_val, "config": config, "split_sha256": hashes, "training_history": history,
              "event_controller_state": net.event_controller.state_dict(), "python_rng": random.getstate(),
              "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        if improved:
            save_checkpoint(ck, out / "best.pth")
            print(f"New best epoch={epoch} Val macro-F1={best_val:.4f}", flush=True)
        save_checkpoint(ck, out / "last.pth")
        write_json(history, history_path)
        del ck
    write_json(history, history_path)  # Reconcile interruption between last.pth and JSON.
    best = load_checkpoint(out / "best.pth")
    validate_config(best["config"], config)
    if best["split_sha256"] != hashes:
        raise ValueError("Best checkpoint split mismatch")
    net.load_state_dict(best["model_state_dict"])
    best_epoch, best_val = best["best_epoch"], best["best_val_f1"]
    del best, optimizer, scheduler
    net.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Inference memory must not include Adam/checkpoint storage.
    best_hash = sha256(out / "best.pth")
    old_hash = read_json(marker).get("best_sha256") if marker.exists() else None
    # Missing completion metadata can mean interruption after checkpoint commit
    # but before evaluation; never reuse metrics from an older best checkpoint.
    invalidate = trained or old_hash != best_hash
    if invalidate or not all(p.exists() for p in required[:5]):
        val = evaluate(loaders["val"], net, device, criterion, "Best Val")
        test = evaluate(loaders["test"], net, device, criterion, "Test")
        save_final_evaluation(out, val, test)
    if not a.no_profile_efficiency and (invalidate or not (out / "efficiency_metrics.json").exists()):
        write_json(profile(datasets["test"], loaders["test"], net, device, a), out / "efficiency_metrics.json")
    write_json({"epochs": a.epochs, "best_epoch": best_epoch, "best_val_macro_f1": best_val,
                "best_sha256": best_hash, "dataset": "cpsc2018", "profile_efficiency": not a.no_profile_efficiency}, marker)
    print("COMPLETE:", out, flush=True)


def main():
    run(parse_args())
