"""顺序运行5种子；默认仅两种时长主实验，消融单独选择。"""
import argparse
import fcntl
import subprocess
import sys
from pathlib import Path
from config import add_options,parse_args
from summarize import summarize

def main():
    parser=add_options(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    parser.add_argument('--suite',choices=('main','ablations','all'),default='main')
    parser.add_argument('--include-reset-all',action='store_true')
    parser.add_argument('--ablation-durations',nargs='+',choices=('variable','fixed'),default=['variable'],
                        help='默认做可变时长消融；传 variable fixed 同时检查固定时长的迹/事件策略')
    project=Path(__file__).resolve().parent
    parser.add_argument('--out-root',default=str(project/'runs'))
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:parser.error('训练种子必须非负且不重复')
    root=Path(args.out_root).resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    if args.suite in ('main','all'):jobs.extend((d,'main') for d in ('variable','fixed'))
    if args.suite in ('ablations','all'):
        for duration in dict.fromkeys(args.ablation_durations):
            variants=('reset_neurons','all_updates','full_trace','no_trace') if duration=='variable' else ('all_updates','full_trace','no_trace')
            jobs.extend((duration,v) for v in variants)
        if args.include_reset_all:jobs.append(('variable','reset_all'))
    elif args.include_reset_all:parser.error('--include-reset-all requires ablations or all')
    common=[]
    for key in ('data_dir','device','epochs','batch_size','num_workers','torch_threads','window_length',
                'repeat_times','overlap','preprocessing','tail_mode','lr','weight_decay','event_warmup_epochs','event_target_skip','event_min_history',
                'event_emergency_error','log_every_batches'):
        common.extend(('--'+key.replace('_','-'),str(getattr(args,key))))
    for key in ('no_augment','no_time_warp','no_training_power','no_padding_benchmark','no_profile_efficiency'):
        if getattr(args,key):common.append('--'+key.replace('_','-'))
    with (root/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        total=len(jobs)*len(args.seeds);index=0
        for seed in args.seeds:
            for duration,variant in jobs:
                index+=1;out=root/f'seed{seed}'/f'{duration}_{variant}'
                options=[*common,'--out-dir',str(out),'--duration',duration,'--variant',variant,
                         '--train-seed',str(seed),'--split-seed',str(seed)]
                parse_args(options)
                log=root/'logs'/f'seed{seed}'/(duration+'_'+variant+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                print(f'[{index}/{total}] train_seed=split_seed={seed} {duration}/{variant}; log={log}',flush=True)
                with log.open('a') as stream:
                    result=subprocess.run([sys.executable,'-u',str(project/'train.py'),*options],stdout=stream,stderr=subprocess.STDOUT)
                if result.returncode:
                    print(log.read_text()[-5000:],flush=True)
                    raise RuntimeError(f'实验失败，停止后续任务。修复后同命令自动续跑：{log}')
                print('Finished (or reused/skipped):',out,flush=True)
            summarize(root)
        summarize(root)
    print('ALL COMPLETE. Summaries:',root/'summary.csv',root/'summary_mean_std.csv',flush=True)

if __name__=='__main__':main()
