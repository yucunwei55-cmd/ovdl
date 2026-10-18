"""One simulation frame on device; CPU sequence stays on host."""
def frame_on_device(data, step, device):
    return data[:, step].to(device, non_blocking=True).permute(0, 2, 1).contiguous()
