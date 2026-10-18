"""One graph per current step; skipped steps still evolve LIF and CHTrace.

The event is decided AFTER the one grad-enabled forward, so activation memory is
not eliminated on skipped steps. No second forward, BN update, or dropout draw.
"""
import json
import time
from collections import Counter
from metrics import compute_classification_metrics, print_classification_metrics
from sklearn.metrics import classification_report, confusion_matrix
from io_utils import write_json
from benchmark import NVMLPowerMeter
import numpy as np
import torch
from torch.nn import functional as F
from spikingjelly.activation_based import functional
from event_controller import EventController, FeatureCapture, trace_rms
from online_trace import trace_cache_stats
from input_stream import frame_on_device


def train_event(dataloader, net, criterion, epoch, scheduler, optimizer, device, args,
                metrics_fn=compute_classification_metrics, print_metrics=print_classification_metrics,
                power_cls=NVMLPowerMeter):
    print(f'\nTraining (cross-window CHTrace / {args.update_policy}) epoch {epoch}:', flush=True)
    net.train()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    power = None
    if device.type == 'cuda' and not args.no_training_power:
        power = power_cls(device_index=device.index if device.index is not None else torch.cuda.current_device(),
                          sample_ms=args.power_sample_ms,
                          device_uuid=getattr(torch.cuda.get_device_properties(device), 'uuid', None))
        if power.available:
            power.start()
    start = time.perf_counter()
    net.gate_sum = 0.
    net.gate_count = 0
    capture = FeatureCapture(net)
    controller = getattr(net, 'event_controller', None)
    if controller is None:
        controller = net.event_controller = EventController(args)
    if hasattr(net, 'reset_boundary_statistics'): net.reset_boundary_statistics()
    reasons = Counter()
    primary_reasons = Counter()
    resources = Counter()
    class_opportunities = np.zeros(4, dtype=np.int64)
    class_updated = np.zeros(4, dtype=np.int64)
    class_events = np.zeros(4, dtype=np.int64)
    peak_trace = peak_dense = peak_controller = 0
    max_skip = 0
    loss_total = 0.
    sample_count = correct_count = 0
    ys, predictions, probabilities = [], [], []
    try:
        for batch_index, (data, labels) in enumerate(dataloader):
            labels = labels.to(device)
            functional.reset_net(net)  # New records, never at a window boundary.
            controller.reset()
            total_steps = data.shape[1]
            out_sum = None
            batch_loss = 0.
            batch_counts = torch.bincount(labels, minlength=4).cpu().numpy()
            for step in range(total_steps):
                optimizer.zero_grad(set_to_none=True)
                x_t = frame_on_device(data, step, device)
                out = net(x_t)
                # 即时 CE：每一步单独反传；最终分类仍使用整条记录的平均 logits。
                loss = criterion(out, labels)
                # All policy control data are detached and current/past only.
                update, detail = controller.decide(out.detach(), labels, capture.features,
                                                  capture.head_input, trace_rms(net), step, epoch)
                capture.features = capture.head_input = None
                cache = trace_cache_stats(net)
                peak_trace = max(peak_trace, cache['historical_trace_cache_bytes'])
                peak_dense = max(peak_dense, cache['full_resolution_equivalent_bytes'])
                peak_controller = max(peak_controller, controller.summary.numel() * controller.summary.element_size())
                resources['potential_optimizer_steps'] += 1
                resources['processed_simulation_steps'] += len(labels)
                resources['sample_step_opportunities'] += len(labels)
                class_opportunities += batch_counts
                class_events += detail['per_class_sample_events']
                primary_reasons[detail['primary_reason']] += 1
                resources['batch_score_sum'] += detail['batch_score']
                if detail['threshold'] is not None:
                    resources['threshold_sum'] += detail['threshold']
                    resources['threshold_count'] += 1
                if update:
                    loss.backward()
                    if hasattr(net, 'boundary_logit') and net.boundary_logit.grad is not None:
                        net.carry_gradient_norm_sum += float(net.boundary_logit.grad.detach().norm())
                        net.carry_gradient_updates += 1
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                    optimizer.step()
                    resources['optimizer_steps'] += 1
                    resources['sample_steps_with_batch_update'] += len(labels)
                    class_updated += batch_counts
                    for name in ['boundary','warmup','calibration','timeout','vote','emergency','all_policy']:
                        reasons[name] += int(detail[name])  # Overlapping reasons.
                else:
                    resources['skipped_backward_steps'] += 1
                    # 当前 batch 的当前时间步不反传、不调用 Adam；前向/LIF/BN/q 仍推进。
                    # 不会整条记录或整个 batch 序列都丢弃，也不是神经元级稀疏更新。
                for name in ['error_sum','credit_sum','novelty_sum','sample_events']:
                    resources[name] += detail[name]
                max_skip = max(max_skip, controller.max_observed_skipped)
                batch_loss += float(loss.detach())
                out_sum = out.detach() if out_sum is None else out_sum + out.detach()
                functional.detach_net(net)  # Cut graph, keep numerical states.
                del loss, out, x_t
            output = out_sum / total_steps
            prob = output.softmax(1)
            pred = prob.argmax(1)
            loss_total += batch_loss / total_steps
            correct_count += int((pred == labels).sum())
            sample_count += len(labels)
            ys.append(labels.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            probabilities.append(prob.cpu().numpy())
            functional.reset_net(net)
            if (batch_index + 1) % args.log_every_batches == 0:
                rate = resources['skipped_backward_steps'] / resources['potential_optimizer_steps']
                print(f'  batch={batch_index + 1}/{len(dataloader)} acc={correct_count/sample_count:.4f} '
                      f'updates={resources["optimizer_steps"]} skip={rate:.2%}', flush=True)
    finally:
        capture.close()
        functional.reset_net(net)
    scheduler.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    measured = power.stop() if power is not None and power.available else (
        {'gpu_energy_j': None, 'unavailable_reason': getattr(power, 'unavailable_reason', 'NVML unavailable')}
        if power is not None else {})
    metrics = metrics_fn(np.concatenate(ys), np.concatenate(predictions), np.concatenate(probabilities), 4)
    print_metrics(f'Epoch {epoch}', loss_total / len(dataloader), metrics)
    opportunities = resources['potential_optimizer_steps']
    stats = {
        'epoch': epoch, 'records': sample_count,
        'processed_simulation_steps': resources['processed_simulation_steps'],
        'unique_ecg_windows': resources['processed_simulation_steps'] // args.repeat_times,
        'optimizer_steps': resources['optimizer_steps'], 'backward_calls': resources['optimizer_steps'],
        'potential_optimizer_steps': opportunities,
        'skipped_backward_steps': resources['skipped_backward_steps'],
        'update_skip_fraction': resources['skipped_backward_steps'] / opportunities,
        'event_reasons_overlapping_batch_counts': dict(reasons),
        'event_primary_reason_batch_counts': dict(primary_reasons),
        'event_target_skip_fraction': args.event_target_skip,
        'mean_batch_event_score': resources['batch_score_sum']/opportunities,
        'mean_lagged_event_threshold': resources['threshold_sum']/resources['threshold_count'] if resources['threshold_count'] else None,
        'quantile_history_entries': sum(len(q) for q in controller.scores.values()),
        'quantile_history_float64_payload_bytes': 8*sum(len(q) for q in controller.scores.values()),
        'cross_window_state': net.boundary_report() if hasattr(net,'boundary_report') else None,
        'mean_correct_class_probability_error': resources['error_sum'] / resources['sample_step_opportunities'],
        'mean_credit_proxy': resources['credit_sum'] / resources['sample_step_opportunities'],
        'mean_history_novelty': resources['novelty_sum'] / resources['sample_step_opportunities'],
        'per_class_sample_step_opportunities': class_opportunities.tolist(),
        'per_class_sample_steps_with_batch_update': class_updated.tolist(),
        'per_class_sample_events': class_events.tolist(),
        'max_consecutive_skipped_steps_in_batch_sequence': max_skip,
        'controller_history_peak_bytes': peak_controller,
        'historical_trace_cache_peak_bytes': peak_trace,
        'full_resolution_trace_equivalent_peak_bytes': peak_dense,
        'historical_cache_reduction_fraction': 1 - peak_trace / peak_dense if peak_dense else None,
        'boundary_gate_mean': float(net.gate_sum / net.gate_count) if net.gate_count else None,
        'boundary_gate_sample_count': net.gate_count,
        'epoch_wall_time_s': elapsed,
        'optimizer_steps_per_epoch': resources['optimizer_steps'],
        'input_storage_policy': 'host_sequence_single_frame_device',
        'throughput_records_per_s_including_dataloader': sample_count / elapsed,
        'peak_cuda_memory_allocated_mb': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None,
        'peak_cuda_memory_reserved_mb': torch.cuda.max_memory_reserved(device)/1024**2 if device.type == 'cuda' else None,
        'gpu_energy_j': measured.get('gpu_energy_j'),
        'gpu_energy_j_per_record': measured.get('gpu_energy_j')/sample_count if measured.get('gpu_energy_j') is not None else None,
        'training_power': measured,
        'training_energy_scope': 'Whole GPU epoch, including loading/forward/controller/backward/optimizer; excludes validation; no idle subtraction',
        'classification': metrics,
        'skip_scope': 'Whole-batch backward and Adam calls; ALL forwards and BN/trace updates still execute',
    }
    print(f'Event summary: updates={stats["optimizer_steps"]}/{opportunities} '
          f'skip={stats["update_skip_fraction"]:.2%} wall={elapsed:.2f}s max_gap={max_skip}', flush=True)
    print('Event primary reasons:',dict(primary_reasons),flush=True)
    if stats['cross_window_state'] is not None:
        print('Selective cross-window state:',json.dumps(stats['cross_window_state']),flush=True)
    return stats


@torch.no_grad()
def sequence_logits(net,data,reset_neurons=False):
    functional.reset_net(net)
    net.intervention_reset_history=reset_neurons
    try:
        device=next(net.parameters()).device
        total=None
        for step in range(data.shape[1]):
            x_t=frame_on_device(data,step,device)
            out=net(x_t);total=out if total is None else total+out
            del x_t,out
        return total/data.shape[1]
    finally:
        functional.reset_net(net)
        net.intervention_reset_history=False

@torch.no_grad()
def evaluate(loader,net,device,criterion,tag='Val',output_path=None):
    net.eval();ys=[];ps=[];loss=0.
    for data,labels in loader:
        labels_device=labels.to(device)
        logits=sequence_logits(net,data)
        loss+=float(criterion(logits,labels_device))
        ys.append(labels.numpy());ps.append(logits.softmax(1).cpu().numpy())
    y=np.concatenate(ys);prob=np.concatenate(ps);pred=prob.argmax(1)
    metrics=compute_classification_metrics(y,pred,prob,4)
    cm=confusion_matrix(y,pred,labels=[0,1,2,3])
    normalized=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)!=0)
    result={'records':len(y),'loss_mean_batches':loss/len(loader),'metrics':metrics,
            'classification_report':classification_report(y,pred,labels=[0,1,2,3],target_names=['N','A','O','~'],output_dict=True,zero_division=0),
            'confusion_matrix':cm.tolist(),'confusion_matrix_row_normalized':normalized.tolist()}
    print_classification_metrics(tag,loss/len(loader),metrics)
    if output_path:
        write_json(result,output_path)
        print(classification_report(y,pred,labels=[0,1,2,3],target_names=['N','A','O','~'],digits=4,zero_division=0))
        print('Confusion matrix N/A/O/~:',cm.tolist(),flush=True)
    return result
