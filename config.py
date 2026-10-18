"""只暴露运行/数据/事件参数；骨干和 V2 学习规则保持不变。"""
import argparse
from pathlib import Path

VARIANTS = ('main', 'reset_neurons', 'all_updates', 'full_trace', 'no_trace', 'reset_all')
CORE_KEYS = ('method duration window_length repeat_times overlap trace_ratio trace_decay trace_tile '
             'trace_backward gate_strength gate_floor loss_mode aux_loss_weight split_seed train_seed '
             'tail_padding no_time_warp reset_at_window leads num_classes lr batch_size num_workers '
             'grad_clip update_policy event_warmup_epochs event_target_skip event_queue_size '
             'event_min_history event_error_weight event_novelty_weight event_class_weights '
             'event_emergency_error event_max_skip event_history_decay event_descriptor_dim '
             'event_trace_cap carry_init carry_max carry_novelty_beta carry_scope boundary_mode').split()

def add_options(parser):
    parser.add_argument('--data-dir', required=True, help='包含 training2017/REFERENCE.csv 的目录')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--torch-threads', type=int, default=0)
    parser.add_argument('--window-length', type=int, default=9000)
    parser.add_argument('--repeat-times', type=int, default=3)
    parser.add_argument('--overlap', type=float, default=.2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--no-time-warp', action='store_true', help='Compatibility flag; time warp is always disabled')
    parser.add_argument('--preprocessing', choices=('legacy','corrected'), default='legacy')
    parser.add_argument('--tail-mode', choices=('pad','anchor'), default='pad')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--event-warmup-epochs', type=int, default=10)
    parser.add_argument('--event-target-skip', type=float, default=.30)
    parser.add_argument('--event-min-history', type=int, default=32)
    parser.add_argument('--event-emergency-error', type=float, default=.98)
    parser.add_argument('--no-training-power', action='store_true')
    parser.add_argument('--no-padding-benchmark', action='store_true')
    parser.add_argument('--no-profile-efficiency', action='store_true')
    parser.add_argument('--log-every-batches', type=int, default=50)
    return parser

def finalize(a):
    if a.split_seed is None:
        a.split_seed=a.train_seed
    if a.train_seed < 0 or a.split_seed != a.train_seed:
        raise ValueError('Paired seeds required: split_seed must equal train_seed and be nonnegative')
    a.data_dir=str(Path(a.data_dir).resolve())
    a.out_dir=str(Path(a.out_dir).resolve())
    a.model_revision='event_chtrace_v3_streamed'
    a.no_time_warp=True
    a.data_protocol='cinc2017_postnorm_augmentation_v2' 
    a.method='current' if a.variant=='no_trace' else 'chtrace_gate'
    a.trace_ratio=1 if a.variant=='full_trace' else 8
    a.trace_decay=.5; a.trace_tile=512; a.trace_backward='tiled'
    a.gate_strength=1.; a.gate_floor=.5; a.reset_at_window=False
    a.loss_mode='instant'; a.aux_loss_weight=.1; a.grad_clip=0.
    a.tail_padding='zero'; a.leads='1'; a.num_classes=4
    a.carry_init=.2; a.carry_max=.5; a.carry_novelty_beta=1.; a.carry_scope='layer4'
    a.boundary_mode={'reset_neurons':'reset_neurons','reset_all':'reset'}.get(a.variant,'selective')
    a.update_policy='all' if a.variant=='all_updates' else 'event'
    a.event_queue_size=256; a.event_error_weight=.5; a.event_novelty_weight=.5
    a.event_class_weights=[1.,1.5,1.,1.5]; a.event_max_skip=2
    a.event_history_decay=.8; a.event_descriptor_dim=32; a.event_trace_cap=2.
    a.power_sample_ms=20.
    if a.weight_decay < 0:raise ValueError('weight_decay must be nonnegative')
    if min(a.epochs,a.batch_size,a.window_length,a.repeat_times,a.log_every_batches)<1:
        raise ValueError('epochs/batch/window/repeat/log interval must be positive')
    if min(a.num_workers,a.torch_threads,a.event_warmup_epochs)<0 or not 1<=a.event_min_history<=256:
        raise ValueError('Invalid workers/threads/warmup/history')
    if not 0<=a.overlap<1 or not 0<a.lr or not 0<=a.event_emergency_error<=1:
        raise ValueError('Invalid overlap/lr/emergency threshold')
    if not 0<=a.event_target_skip<=(a.repeat_times-1)/a.repeat_times:
        raise ValueError('Target skip exceeds available non-boundary steps')
    return a

def parse_args(argv=None):
    parser=add_options(argparse.ArgumentParser(description='CinC2017 clean V2'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--duration', choices=('variable','fixed'), default='variable')
    parser.add_argument('--variant', choices=VARIANTS, default='main')
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=None, help='Defaults to train-seed; must match train-seed')
    return finalize(parser.parse_args(argv))

def validate_config(previous,current,legacy=False):
    keys=CORE_KEYS if legacy else CORE_KEYS+['model_revision','variant','torch_threads','weight_decay','data_protocol','preprocessing','tail_mode','no_augment']
    bad=[k for k in keys if previous.get(k)!=current.get(k)]
    if bad: raise ValueError('实验配置不同，拒绝覆盖/复用: '+', '.join(bad))
