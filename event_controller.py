"""V2 事件控制器原样保留；只使用当前/过去信息。

决定的是一个 batch 内某个模拟步骤是否反传，而非丢弃整批记录。
当前评分在决定之后入队，窗口首步和热身阶段强制更新。
Causal adaptive batch update controller; thresholds use past scores only.

The skip target controls a quantile, not a quota or a promised speedup. Boundary
steps always update, so the selective state gates receive a current-loss gradient.
"""
import math
from collections import deque
import numpy as np
import torch
from torch.nn import functional as F
from online_trace import TraceMemory


class FeatureCapture:
    def __init__(self, net):
        self.features = self.head_input = None
        self.handles = [net.dropout.register_forward_pre_hook(self._features),
                        net.fc.register_forward_pre_hook(self._head)]

    def _features(self, module, inputs):
        self.features = inputs[0].detach()

    def _head(self, module, inputs):
        self.head_input = inputs[0].detach()

    def close(self):
        for handle in self.handles: handle.remove()
        self.features = self.head_input = None


@torch.no_grad()
def trace_rms(net):
    memories = [m for m in net.modules() if isinstance(m, TraceMemory) and m.q is not None]
    if not memories: return next(net.parameters()).new_zeros(())
    count = sum(m.q.numel() for m in memories)
    return (torch.stack([m.q.square().sum() for m in memories]).sum() / count).sqrt()


class EventController:
    def __init__(self, args):
        self.args = args
        self.scores = {p: deque(maxlen=args.event_queue_size) for p in range(1,args.repeat_times)}
        self.reset()

    def reset(self):
        # Called for new record batches. Threshold history survives records/epochs.
        self.summary = None
        self.skipped = 0
        self.max_observed_skipped = 0

    def state_dict(self):
        return {'scores': {str(k): list(v) for k,v in self.scores.items()}}

    def load_state_dict(self, state):
        values=state['scores']
        if set(values)!=set(str(k) for k in self.scores):
            raise ValueError('Controller checkpoint repeat-times mismatch')
        for k,q in self.scores.items():
            entries=values[str(k)]
            if len(entries)>q.maxlen or not all(np.isfinite(v) and v>=0 for v in entries):
                raise ValueError('Invalid controller score history')
            q.clear();q.extend(entries)
        self.reset()

    @torch.no_grad()
    def decide(self, logits, labels, features, head_input, historical_rms, step, epoch):
        a=self.args;position=step % a.repeat_times
        # 当前特征相对过去 EMA 的变化；此处是样本评分，最终按整个 batch 决策。
        descriptor=F.adaptive_avg_pool1d(features.unsqueeze(1),a.event_descriptor_dim).squeeze(1)
        if self.summary is None:
            novelty=descriptor.new_zeros(len(descriptor))
        else:
            scale=(descriptor.square().sum(1)+self.summary.square().sum(1)).sqrt().clamp_min(1e-6)
            novelty=(descriptor-self.summary).norm(dim=1)/scale
        self.summary=descriptor.clone() if self.summary is None else (
            a.event_history_decay*self.summary+(1-a.event_history_decay)*descriptor)
        probability=logits.softmax(1)
        error=1-probability.gather(1,labels[:,None]).squeeze(1)
        delta=probability-F.one_hot(labels,logits.shape[1]).to(probability.dtype)
        # Current fc CE gradient proxy, excluding bias/batch scaling; no claim
        # that it equals the historical or full backbone gradient.
        head_proxy=delta.norm(dim=1)*head_input.norm(dim=1)/math.sqrt(head_input.shape[1])
        credit=head_proxy*(1+historical_rms.clamp(max=a.event_trace_cap))
        sample_score=torch.log1p(credit)+a.event_error_weight*error+a.event_novelty_weight*novelty
        class_weight=sample_score.new_tensor(a.event_class_weights)[labels]
        batch_score=float((sample_score*class_weight).sum()/class_weight.sum())
        boundary=position==0;warmup=epoch<=a.event_warmup_epochs
        queue=self.scores.get(position)
        calibration=not boundary and len(queue)<a.event_min_history
        # Mandatory window-boundary updates occupy 1/repeat_times of steps.
        free_fraction=(a.repeat_times-1)/a.repeat_times
        quantile=min(1.,a.event_target_skip/free_fraction) if free_fraction else 0.
        # 阈值仅来自过去队列；本步评分尚未入队，不看未来或当前分数计算阈值。
        threshold=float(np.quantile(queue,quantile)) if not boundary and not calibration else None
        vote=a.event_target_skip==0 or (threshold is not None and batch_score>threshold)
        timeout=self.skipped>=a.event_max_skip
        emergency=bool((error>a.event_emergency_error).any())
        # 任一强制条件成立就更新；边界强制更新保证主方法的状态门获得梯度。
        update=a.update_policy=='all' or boundary or warmup or calibration or timeout or vote or emergency
        # Append AFTER deciding: no current/future score enters this threshold.
        if queue is not None: queue.append(batch_score)
        self.skipped=0 if update else self.skipped+1
        self.max_observed_skipped=max(self.max_observed_skipped,self.skipped)
        sample_events=sample_score>threshold if threshold is not None else torch.ones_like(labels,dtype=torch.bool)
        detail={'boundary':boundary,'warmup':warmup,'calibration':calibration,
                'timeout':timeout,'vote':vote,'emergency':emergency,'all_policy':a.update_policy=='all',
                'batch_score':batch_score,'threshold':threshold,'free_step_skip_quantile':quantile,
                'error_sum':float(error.sum()),'credit_sum':float(credit.sum()),
                'novelty_sum':float(novelty.sum()),'trace_rms':float(historical_rms),
                'sample_events':int(sample_events.sum()),
                'per_class_sample_events':torch.bincount(labels[sample_events],minlength=4).cpu().tolist()}
        # Mutually exclusive reasons allow identification of the dominant force.
        priority=['all_policy','warmup','boundary','calibration','emergency','timeout','vote']
        detail['primary_reason']=next((k for k in priority if detail[k]),'skip')
        return bool(update),detail
