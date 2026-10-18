"""训练测量和推理测量分离；分组加速不能归因于事件跳过。"""
import os
import time
import threading
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional,neuron
from sampling import seed_worker
from input_stream import frame_on_device

class NVMLPowerMeter:
    """NVML 全 GPU 实测功率积分；不等价于进程独占能耗。"""
    def __init__(self, device_index=0, sample_ms=20.0, device_uuid=None):
        self.available = False
        self.sample_s = max(float(sample_ms) / 1000.0, 0.005)
        self.samples = []
        self._stop_evt = threading.Event()
        self._thread = None
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            if device_uuid:
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(device_uuid))
            else:
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                token = visible.split(',')[int(device_index)].strip() if visible else str(device_index)
                if token.startswith(('GPU-', 'MIG-')):
                    self.handle = pynvml.nvmlDeviceGetHandleByUUID(token)
                elif token.isdigit():
                    self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(token))
                else:
                    raise ValueError('Cannot resolve logical CUDA device to an NVML GPU')
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode(errors='ignore')
            self.available = True
        except Exception as exc:
            self.available = False
            self.unavailable_reason = str(exc)

    def _worker(self):
        while not self._stop_evt.is_set():
            ts = time.perf_counter()
            try:
                p = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((ts, float(p)))
            except Exception:
                pass
            self._stop_evt.wait(self.sample_s)

    def start(self):
        if not self.available:
            return
        self.samples = []
        self._stop_evt.clear()
        try:
            self.samples.append((time.perf_counter(), self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0))
        except Exception:
            pass
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available:
            return {}
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            self.samples.append((time.perf_counter(), self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0))
        except Exception:
            pass
        if len(self.samples) < 2:
            return {
                'gpu_name': self.gpu_name,
                'mean_power_w': None,
                'gpu_energy_j': None,
            }
        t = np.asarray([x[0] for x in self.samples], dtype=np.float64)
        p = np.asarray([x[1] for x in self.samples], dtype=np.float64)
        duration = float(t[-1] - t[0])
        _trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        energy_j = float(_trap(p, t))
        mean_power = float(energy_j / duration) if duration > 0 else float(p.mean())
        return {
            'gpu_name': self.gpu_name,
            'mean_power_w': mean_power,
            'gpu_energy_j': energy_j,
            'power_sample_count': int(len(self.samples)),
            'power_sampling_quality': 'short_run_few_samples' if len(self.samples) < 10 else 'sampled',
            'power_integrated_duration_s': duration,
        }

def estimate_conv_linear_macs(net, nleads, signal_length, device):
    """Count Conv1d/Linear MACs for one sample and one simulation step."""
    total_macs = {'value': 0}
    handles = []

    def conv_hook(m, inp, out):
        if not torch.is_tensor(out):
            return
        batch = int(out.shape[0])
        out_ch = int(out.shape[1])
        out_len = int(out.shape[2])
        kernel = int(m.kernel_size[0])
        cin_per_group = int(m.in_channels // m.groups)
        total_macs['value'] += batch * out_ch * out_len * kernel * cin_per_group

    def linear_hook(m, inp, out):
        if not torch.is_tensor(out):
            return
        total_macs['value'] += int(out.numel()) * int(m.in_features)

    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = net.training
    try:
        net.eval()
        functional.reset_net(net)
        with torch.no_grad():
            dummy = torch.zeros((1, int(nleads), int(signal_length)), device=device)
            _ = net(dummy)
    finally:
        functional.reset_net(net)
        for h in handles:
            h.remove()
        net.train(was_training)
    macs = int(total_macs['value'])
    return {
        'conv_linear_macs_per_simulation_step': macs,
        'conv_linear_flops_per_simulation_step_approx_2xmac': int(2 * macs),
        'note': 'Counts Conv1d and Linear only; BN/pooling/activation/control overhead is excluded.'
    }

def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def padded_collate(batch):
    data,labels=zip(*batch);lengths=[x.shape[0] for x in data]
    padded=data[0].new_zeros((len(data),max(lengths),*data[0].shape[1:]))
    for index,x in enumerate(data): padded[index,:x.shape[0]]=x
    return padded,torch.stack(labels),lengths

@torch.no_grad()
def run_sequence(net,data):
    functional.reset_net(net)
    try:
        device=next(net.parameters()).device
        for t in range(data.shape[1]):
            x_t=frame_on_device(data,t,device)
            net(x_t)
            del x_t
    finally: functional.reset_net(net)

@torch.no_grad()
def benchmark(loader,net,device,args,padded=False):
    net.eval()
    # 同一 batch 预热3次；预热和 H2D 均不计入模型延迟。
    first=next(iter(loader))[0].to(device)
    for _ in range(3): run_sequence(net,first)
    del first
    sync(device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    records=batches=real=processed=0;elapsed=0.;energy=0.;valid_energy=True;energy_samples=0
    power=NVMLPowerMeter(device_index=device.index or 0,sample_ms=args.power_sample_ms,
                         device_uuid=getattr(torch.cuda.get_device_properties(device),'uuid',None)) if device.type=='cuda' else None
    for batch in loader:
        data=batch[0].to(device);lengths=batch[2] if padded else [data.shape[1]]*len(data)
        sync(device)
        if power is not None and power.available: power.start()
        start=time.perf_counter();run_sequence(net,data);sync(device);elapsed+=time.perf_counter()-start
        measured=power.stop() if power is not None and power.available else {}
        if measured.get('gpu_energy_j') is None: valid_energy=False
        else: energy+=measured['gpu_energy_j'];energy_samples+=measured.get('power_sample_count',0)
        records+=len(data);batches+=1;real+=sum(lengths);processed+=len(data)*data.shape[1]
    return {'records':records,'batches':batches,'real_simulation_steps':real,
            'processed_simulation_steps':processed,'ecg_windows':real//args.repeat_times,
            'temporal_overhead':processed/real,'model_time_s':elapsed,
            'latency_ms_per_record':1000*elapsed/records,'throughput_records_per_s':records/elapsed,
            'peak_cuda_memory_allocated_mb':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
            'peak_cuda_memory_reserved_mb':torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None,
            'gpu_energy_j':energy if valid_energy else None,'power_sample_count':energy_samples,
            'energy_scope':'sum of whole-GPU timed batch intervals; short intervals may have coarse NVML resolution',
            'runtime_only':padded,
            'timing_scope':'whole sequence H2D excluded; frame permutation included',
            'input_storage_policy':'single reordered frame; raw GPU sequence resident'}

@torch.no_grad()
def firing_fraction(loader,net,device):
    counts=[0,0];handles=[]
    def hook(module,inputs,output):
        counts[0]+=int(torch.count_nonzero(output));counts[1]+=output.numel()
    for module in net.modules():
        if isinstance(module,neuron.BaseNode): handles.append(module.register_forward_hook(hook))
    try:
        for data,_ in loader: run_sequence(net,data.to(device))
    finally:
        for h in handles:h.remove()
        functional.reset_net(net)
    return counts[0]/counts[1] if counts[1] else None

def profile(dataset,loader,net,device,args):
    macs=estimate_conv_linear_macs(net,1,args.window_length,device)
    grouped=benchmark(loader,net,device,args)
    result={'parameters':sum(p.numel() for p in net.parameters()),
            'parameter_storage_mb':sum(p.numel()*p.element_size() for p in net.parameters())/1024**2,
            'operations':macs,'grouped_inference':grouped,
            'overall_lif_firing_fraction':firing_fraction(loader,net,device),
            'measurement_note':'推理不执行 backward/Adam；事件策略只影响训练，分组收益单独报告。MAC 不是实测能耗。'}
    macs['conv_linear_macs_per_record_mean']=macs['conv_linear_macs_per_simulation_step']*grouped['real_simulation_steps']/grouped['records']
    if not args.no_padding_benchmark:
        padding=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                           pin_memory=True,collate_fn=padded_collate,worker_init_fn=seed_worker,
                           generator=torch.Generator().manual_seed(args.train_seed))
        result['padding_inference']=benchmark(padding,net,device,args,padded=True)
        result['grouping_latency_speedup']=result['padding_inference']['latency_ms_per_record']/grouped['latency_ms_per_record']
    print('Inference:',grouped,flush=True)
    return result
