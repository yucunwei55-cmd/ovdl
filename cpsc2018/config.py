"""CPSC2018 entry configuration; V2 presets are shared with CinC2017."""
import argparse
from pathlib import Path
from config import VARIANTS, finalize, validate_config as validate_core
from .data import LEADS


def parser(single=True):
    p = argparse.ArgumentParser(description="OVDL V2 on CPSC2018")
    p.add_argument("--data-dir", required=True, help="500 Hz WFDB record directory (.hea + signal files)")
    p.add_argument("--device", default="cuda:0")
    for name, kind, default in (("epochs", int, 100), ("batch-size", int, 32),
                                ("num-workers", int, 4), ("torch-threads", int, 0),
                                ("window-length", int, 9000), ("repeat-times", int, 4),
                                ("overlap", float, .2), ("lr", float, 1e-4),
                                ("event-warmup-epochs", int, 10), ("event-target-skip", float, .30),
                                ("event-min-history", int, 32), ("event-emergency-error", float, .98),
                                ("log-every-batches", int, 50)):
        p.add_argument("--" + name, type=kind, default=default)
    for name in ("no-training-power", "no-padding-benchmark", "no-profile-efficiency"):
        p.add_argument("--" + name, action="store_true")
    p.add_argument("--labels", default="", help="CSV with patient_id/fold and all nine multi-hot diagnoses, or full original diagnosis slots")
    p.add_argument("--reference-csv", default="", help="Original CSV with all three diagnosis slots")
    p.add_argument("--leads", default="all", help="all or names such as I,II,V1")
    p.add_argument("--split-mode", choices=("folds", "stratified"), default="folds")
    p.add_argument("--expected-af-count", type=int, default=1221, help="Any-diagnosis AF count; 0 disables audit for intentional subsets")
    p.add_argument("--preprocessing", choices=("corrected", "legacy"), default="legacy")
    p.add_argument("--tail-mode", choices=("anchor", "pad"), default="pad")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0., help="Matched baseline; previous OVDL uses 0.0005")
    if single:
        p.add_argument("--split-seed", type=int, default=None, help="Defaults to train-seed; must match train-seed")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--duration", choices=("variable", "fixed"), default="variable")
        p.add_argument("--variant", choices=VARIANTS, default="main")
        p.add_argument("--train-seed", type=int, default=42)
    else:
        project = Path(__file__).resolve().parents[1]
        p.add_argument("--out-root", default=str(project / "runs/cpsc2018_af_priority_v2"))
        p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
        p.add_argument("--durations", nargs="+", choices=("variable", "fixed"), default=["fixed", "variable"])
        p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["main"])
    return p


def parse_args(argv=None):
    a = parser().parse_args(argv)
    if a.split_seed is None:
        a.split_seed = a.train_seed
    leads = a.leads
    selected = LEADS if leads == "all" else leads.split(",")
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(LEADS):
        raise ValueError("Use all or standard lead names, e.g. I,II,V1")
    if a.overlap != .2:
        raise ValueError("CPSC2018 reference preprocessing uses 20% overlap")
    if a.expected_af_count < 0 or a.weight_decay < 0 or min(a.train_seed, a.split_seed) < 0:
        raise ValueError("Invalid count/seed/weight decay")
    a = finalize(a)
    a.dataset = "cpsc2018"
    a.worker_start_method = "spawn"
    a.data_protocol = "af_priority_v4_postnorm_" + a.preprocessing + "_" + a.tail_mode
    a.num_classes = 3
    a.leads = leads
    a.labels = str(Path(a.labels).resolve()) if a.labels else ""
    a.reference_csv = str(Path(a.reference_csv).resolve()) if a.reference_csv else None
    a.no_time_warp = True  # CPSC reference augmentation has scaling/shift, no time warp.
    a.event_class_weights = [1., 1.5, 1.]
    return a


def validate_config(previous, current):
    validate_core(previous, current)
    keys = ["dataset", "data_protocol", "split_mode", "expected_af_count", "no_augment", "weight_decay", "worker_start_method", "preprocessing", "tail_mode"]
    bad = [key for key in keys if previous.get(key) != current.get(key)]
    if bad:
        raise ValueError("CPSC2018 configuration mismatch: " + ", ".join(bad))
