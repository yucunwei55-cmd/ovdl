"""Sequential OVDL CPSC2018 experiments; five paired train/split seeds by default."""
import fcntl
import subprocess
import sys
from pathlib import Path
from .config import parser, parse_args
from .summarize import summarize


def main(argv=None):
    args = parser(single=False).parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be nonnegative and unique")
    if len(set(args.durations)) != len(args.durations) or len(set(args.variants)) != len(args.variants):
        raise ValueError("Do not repeat durations/variants")
    project = Path(__file__).resolve().parents[1]
    root = Path(args.out_root).resolve()
    data = Path(args.data_dir).resolve()
    common = ["--data-dir", str(data)]
    for key in ("device", "epochs", "batch_size", "num_workers", "torch_threads", "window_length", "repeat_times",
                "overlap", "lr", "leads", "preprocessing", "tail_mode", "split_mode", "expected_af_count", "weight_decay",
                "event_warmup_epochs", "event_target_skip", "event_min_history", "event_emergency_error", "log_every_batches"):
        common.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    if args.labels:
        common.extend(("--labels", str(Path(args.labels).resolve())))
    if args.reference_csv:
        common.extend(("--reference-csv", str(Path(args.reference_csv).resolve())))
    for key in ("no_augment", "no_training_power", "no_padding_benchmark", "no_profile_efficiency"):
        if getattr(args, key):
            common.append("--" + key.replace("_", "-"))
    # Validate all configurations before scheduling any training.
    jobs = []
    for seed in args.seeds:
        for duration in args.durations:
            for variant in args.variants:
                if duration == "fixed" and variant in ("reset_neurons", "reset_all"):
                    print("SKIP redundant fixed boundary-reset preset:", variant, flush=True)
                    continue
                out = root / f"seed{seed}" / (duration + "_" + variant)
                options = [*common, "--out-dir", str(out), "--train-seed", str(seed),
                           "--split-seed", str(seed), "--duration", duration, "--variant", variant]
                parse_args(options)
                jobs.append((seed, duration, variant, out, options))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, (seed, duration, variant, out, options) in enumerate(jobs, 1):
            log = root / "logs" / f"seed{seed}" / (duration + "_" + variant + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(jobs)}] CPSC2018 train_seed=split_seed={seed} {duration}/{variant}; log={log}", flush=True)
            with log.open("a") as stream:
                result = subprocess.run([sys.executable, "-u", str(project / "train_cpsc2018.py"), *options],
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-5000:], flush=True)
                raise RuntimeError("Experiment failed; fix and rerun the same command: " + str(log))
            summarize(root)
            print("Finished (or resumed/skipped):", out, flush=True)
    print("ALL COMPLETE:", root / "summary_mean_std.csv", flush=True)
