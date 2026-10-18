"""历史迹：当前 forward 不变，只在指定卷积的权重梯度中使用旧 q。

Current-preserving historical traces for Conv1d weight gradients.

Forward and input gradients use the actual activity x. Weight gradients use
x + decay * gate * U(q_previous). q is detached, spatially block-averaged history.
No history enters the forward value. This is an OTTT-style backbone adaptation,
not an exact BPTT gradient, and compression is an additional approximation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import base


def compress_blocks(x, ratio):
    """Mean of non-overlapping blocks; last partial block excludes padding."""
    if ratio == 1:
        return x.detach().clone()
    length = x.shape[-1]
    cells = math.ceil(length / ratio)
    padded = F.pad(x, (0, cells * ratio - length))
    blocks = padded.reshape(*x.shape[:-1], cells, ratio)
    counts = x.new_full((cells,), ratio)
    counts[-1] = length - (cells - 1) * ratio
    return blocks.sum(-1) / counts


def history_slice(q, ratio, start, stop):
    """Reconstruct only the requested input positions, never the whole trace."""
    indices = torch.arange(start, stop, device=q.device) // ratio
    return q.index_select(-1, indices)


class TraceMemory(base.MemoryModule):
    def __init__(self, ratio=8, decay=0.5):
        super().__init__()
        if ratio < 1 or not 0 <= decay < 1:
            raise ValueError('ratio must be >=1 and decay in [0,1)')
        self.ratio = int(ratio)
        self.decay = float(decay)
        self.register_memory('q', None)
        self.last_dense_bytes = 0

    def prepare(self, x, gate):
        # 先取旧 q 再推进缓存：当前步权重梯度只能用 q_(t-1)，不能偷用 q_t。
        """Return the OLD history and replace the cache out-of-place.

        q_t = decay * gate_t * q_(t-1) + P(x_t).
        The backward of this step must see q_(t-1), not q_t.
        """
        with torch.no_grad():
            pooled = compress_blocks(x.detach(), self.ratio)
            previous = self.q
            if previous is not None and (previous.shape != pooled.shape or
                    previous.device != pooled.device or previous.dtype != pooled.dtype):
                raise RuntimeError('Trace shape/device changed without reset_net().')
            self.q = pooled if previous is None else (
                self.decay * gate * previous + pooled)
            self.last_dense_bytes = x.numel() * x.element_size()
        return previous


class _TraceConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, previous, gate, decay, ratio,
                stride, padding, dilation, groups, tile, implementation):
        ctx.save_for_backward(x, weight, previous, gate)
        ctx.settings = (decay, ratio, stride, padding, dilation, groups, tile, implementation)
        ctx.has_bias = bias is not None
        return F.conv1d(x, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, previous, gate = ctx.saved_tensors
        decay, ratio, stride, padding, dilation, groups, tile, implementation = ctx.settings
        grad_output = grad_output.contiguous()
        gx = gw = gb = None
        if ctx.needs_input_grad[1]:
            # 权重梯度用 x + rho * gate * U(q_prev)，forward 数值仍只由当前 x 决定。
            if previous is None or decay == 0:
                gw = torch.nn.grad.conv1d_weight(x, weight.shape, grad_output,
                                               stride, padding, dilation, groups)
            elif implementation == 'materialize':
                effective = x + decay * gate * history_slice(previous, ratio, 0, x.shape[-1])
                gw = torch.nn.grad.conv1d_weight(effective, weight.shape, grad_output,
                                               stride, padding, dilation, groups)
                del effective
            else:
                # Partition output positions. Each tile contains the complete
                # convolution receptive fields of those positions. This is
                # algebraically identical to materializing the history, up to
                # floating-point accumulation, including groups and dilation.
                gw = torch.zeros_like(weight)
                length = x.shape[-1]
                receptive = dilation * (weight.shape[-1] - 1) + 1
                for j0 in range(0, grad_output.shape[-1], tile):
                    j1 = min(j0 + tile, grad_output.shape[-1])
                    lo = j0 * stride - padding
                    hi = (j1 - 1) * stride - padding + receptive
                    left, right = max(lo, 0), min(hi, length)
                    if left < right:
                        effective = x[..., left:right] + decay * gate * history_slice(
                            previous, ratio, left, right)
                        effective = F.pad(effective, (max(0, -lo), max(0, hi - length)))
                    else:
                        effective = x.new_zeros((*x.shape[:-1], hi - lo))
                    gw.add_(torch.nn.grad.conv1d_weight(
                        effective, weight.shape, grad_output[..., j0:j1].contiguous(),
                        stride, 0, dilation, groups))
                    del effective
        # Same gradient formulas; allocate gx only after reconstructed inputs are gone.
        if ctx.needs_input_grad[0]:
            gx = torch.nn.grad.conv1d_input(x.shape, weight, grad_output,
                                          stride, padding, dilation, groups)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = grad_output.sum((0, 2))
        return gx, gw, gb, None, None, None, None, None, None, None, None, None, None


class TraceConv1d(nn.Conv1d):
    """State dict weight/bias keys stay compatible with ordinary Conv1d."""
    def __init__(self, *args, ratio=8, decay=0.5, tile=512,
                 implementation='tiled', **kwargs):
        super().__init__(*args, **kwargs)
        if self.padding_mode != 'zeros' or isinstance(self.padding, str):
            raise ValueError('Only integer zero padding is supported.')
        self.history = TraceMemory(ratio, decay)
        self.tile = int(tile)
        if self.tile < 1 or implementation not in ('tiled', 'materialize'):
            raise ValueError('Invalid trace tile/implementation')
        self.implementation = implementation
        self.gate = None

    def forward(self, x):
        # Inference never allocates/updates historical-gradient caches.
        if not self.training:
            return super().forward(x)
        gate = self.gate
        if gate is None:
            gate = x.new_ones((x.shape[0], 1, 1))
        previous = self.history.prepare(x, gate)
        return _TraceConv.apply(x, self.weight, self.bias, previous, gate,
                                self.history.decay, self.history.ratio,
                                self.stride[0], self.padding[0], self.dilation[0],
                                self.groups, self.tile, self.implementation)


def trace_cache_stats(net):
    memories = [m for m in net.modules() if isinstance(m, TraceMemory)]
    return {
        'historical_trace_cache_bytes': sum(m.q.numel() * m.q.element_size()
                                            for m in memories if m.q is not None),
        'full_resolution_equivalent_bytes': sum(m.last_dense_bytes for m in memories
                                                if m.q is not None),
        'trace_sites': len(memories),
    }
