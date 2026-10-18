"""Use the shared profiler with the real CPSC input channel count."""
import torch
from torch.utils.data import DataLoader
from benchmark import estimate_conv_linear_macs, benchmark, firing_fraction, padded_collate
from sampling import seed_worker


def profile(dataset, loader, net, device, args):
    # CinC2017's original profile assumes one input channel. Call its lower-level
    # helpers with dataset.nleads, without editing the working CinC2017 code.
    operations = estimate_conv_linear_macs(net, dataset.nleads, args.window_length, device)
    grouped = benchmark(loader, net, device, args)
    operations["conv_linear_macs_per_record_mean"] = (
        operations["conv_linear_macs_per_simulation_step"] * grouped["real_simulation_steps"] / grouped["records"])
    result = {"input_channels": dataset.nleads, "leads": dataset.selected_leads,
              "parameters": sum(p.numel() for p in net.parameters()),
              "parameter_storage_mb": sum(p.numel() * p.element_size() for p in net.parameters()) / 1024**2,
              "operations": operations, "grouped_inference": grouped,
              "overall_lif_firing_fraction": firing_fraction(loader, net, device),
              "measurement_note": "Inference executes no backward/Adam; padding comparison is runtime-only"}
    if not args.no_padding_benchmark:
        padding = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=padded_collate, worker_init_fn=seed_worker,
                             generator=torch.Generator().manual_seed(args.train_seed),
                             **({"multiprocessing_context": "spawn"} if args.num_workers else {}))
        result["padding_inference"] = benchmark(padding, net, device, args, padded=True)
        result["grouping_latency_speedup"] = (result["padding_inference"]["latency_ms_per_record"] /
                                             grouped["latency_ms_per_record"])
    print("Inference:", grouped, flush=True)
    return result
