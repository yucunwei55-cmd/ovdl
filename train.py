"""单实验入口：自动恢复 epoch 边界；完成后跳过训练与已完成分析。"""
import fcntl
import os
import random
import platform
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from config import parse_args,validate_config
from data import ECGDataset
from sampling import TimeStepGroupSampler,custom_collate_fn,seed_everything,seed_worker
from model import from_config
from event_controller import EventController
from engine import train_event,evaluate
from benchmark import profile
from state_ablation import evaluate_state
from io_utils import read_json,write_json,make_splits,load_checkpoint,save_checkpoint,sha256


def build_loaders(a):
    datasets={};loaders={}
    options=dict(timestep_length=a.window_length,repeat_times=a.repeat_times,
                 overlap_ratio=a.overlap,tail_padding=a.tail_padding,
                 enable_time_warp=False,leads=a.leads,duration=a.duration,
                 preprocessing=a.preprocessing,tail_mode=a.tail_mode,seed=a.train_seed,augment=not a.no_augment)
    for name in ('train','val','test'):
        datasets[name]=ECGDataset(phase='train' if name=='train' else 'test',
            data_dir=str(Path(a.data_dir)/'training2017'),label_csv=str(Path(a.out_dir)/'splits'/(name+'.csv')),**options)
    # Infer channels without consuming global augmentation RNG.
    channels=datasets['train'][0][0].shape[-1]
    generator=torch.Generator().manual_seed(a.train_seed)
    for name in ('train','val','test'):
        ds=datasets[name];sampler=TimeStepGroupSampler(ds.time_steps_list,a.batch_size,shuffle=name=='train')
        loaders[name]=DataLoader(ds,batch_sampler=sampler,num_workers=a.num_workers,pin_memory=True,
            collate_fn=custom_collate_fn,worker_init_fn=seed_worker,generator=generator)
    return datasets,loaders,generator,channels


def run(a):
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    # 同一路径只允许一个进程写入，避免两次 nohup 同时破坏 checkpoint。
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(a,out)


def execute(a,out):
    if a.torch_threads:torch.set_num_threads(a.torch_threads)
    seed_everything(a.train_seed)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('请求 CUDA 但不可用；不会悄悄改成 CPU 训练')
    config=vars(a).copy();metadata=out/'run_config.json';prior=read_json(metadata) if metadata.exists() else None
    if prior:validate_config(prior['config'],config)
    paths,hashes,reference_hash=make_splits(a.data_dir,a.split_seed,out/'splits')
    if prior and (prior['split_sha256']!=hashes or prior['reference_sha256']!=reference_hash):
        raise ValueError('REFERENCE/划分内容与已保存实验不同')
    history=read_json(out/'training_history.json') if (out/'training_history.json').exists() else []
    completed_epoch=history[-1]['epoch'] if history else 0
    if completed_epoch>a.epochs:raise ValueError('已训练 epoch 大于所请求值，拒绝覆盖')
    state_required=a.duration=='variable' and a.variant=='main'
    required=[out/'classification_metrics.json']
    if not a.no_profile_efficiency:required.append(out/'efficiency_metrics.json')
    if state_required:required.extend(out/'state_ablation'/s/'state_ablation.json' for s in ('val','test'))
    marker=out/'complete.json'
    if completed_epoch==a.epochs and marker.exists() and (out/'best.pth').exists() and all(x.exists() for x in required):
        key=read_json(marker)
        if key.get('best_sha256')==sha256(out/'best.pth') and key.get('epochs')==a.epochs:
            print('SKIP completed:',out,flush=True);return
    meta={'config':config,'split_sha256':hashes,'reference_sha256':reference_hash,
          'split_unit':'record','patient_split':False,'class_mapping':{'N':0,'A':1,'O':2,'~':3},
          'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                         'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                         'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}}
    if prior and 'imported_v2' in prior:meta['imported_v2']=prior['imported_v2']
    write_json(meta,metadata)
    datasets,loaders,generator,channels=build_loaders(a)
    write_json({name:{'records':len(ds),'ecg_window_count_histogram':dict(sorted(Counter(ds.time_steps_list).items())),
                      'total_simulation_steps':sum(ds.time_steps_list)*a.repeat_times,
                      'batches':len(loaders[name])} for name,ds in datasets.items()},out/'duration_stats.json')
    net=from_config(config,channels,device);net.event_controller=EventController(a)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=25,gamma=.5)
    best_val=-1.;best_epoch=-1;first=1
    if (out/'last.pth').exists():
        ck=load_checkpoint(out/'last.pth',device);validate_config(ck['config'],config)
        if ck['split_sha256']!=hashes:raise ValueError('Checkpoint split mismatch')
        if ck['epoch']>a.epochs:raise ValueError('Checkpoint 超过请求 epoch，拒绝倒退')
        net.load_state_dict(ck['model_state_dict']);net.event_controller.load_state_dict(ck['event_controller_state'])
        optimizer.load_state_dict(ck['optimizer_state_dict']);scheduler.load_state_dict(ck['scheduler_state_dict'])
        best_val=ck['best_val_f1'];best_epoch=ck['best_epoch'];first=ck['epoch']+1
        history=ck['training_history']
        # 恢复优化器、事件阈值队列和全部 RNG，不能只恢复权重。
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng'].cpu());generator.set_state(ck['loader_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        del ck
        print(f'RESUME at epoch {first} (already {first-1})',flush=True)
    elif history or (out/'best.pth').exists():
        raise ValueError('已有结果却没有 last.pth，拒绝从头覆盖')
    if first<=a.epochs and marker.exists():marker.unlink()
    trained=False
    for epoch in range(first,a.epochs+1):
        trained=True
        datasets['train'].epoch=epoch
        stats=train_event(loaders['train'],net,criterion,epoch,scheduler,optimizer,device,a)
        stats['validation']=evaluate(loaders['val'],net,device,criterion,'Val')
        stats['validation']['selective_state']=net.boundary_report()
        history.append(stats);f1=stats['validation']['metrics']['f1_macro'];improved=f1>best_val
        if improved:best_val=f1;best_epoch=epoch
        ck={'model_state_dict':net.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),'epoch':epoch,'best_epoch':best_epoch,
            'best_val_f1':best_val,'config':config,'split_sha256':hashes,'training_history':history,
            'event_controller_state':net.event_controller.state_dict(),'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'loader_rng':generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        # best 先落盘：中断时旧 last 会重跑当前未提交 epoch，仍能准确恢复。
        if improved:save_checkpoint(ck,out/'best.pth');print(f'New best epoch={epoch} Val macro-F1={best_val:.4f}',flush=True)
        save_checkpoint(ck,out/'last.pth');write_json(history,out/'training_history.json')
        del ck
    # last.pth 是已提交 epoch 的权威记录，修复中断在 checkpoint/JSON 之间的历史缺口。
    write_json(history,out/'training_history.json')
    ck=load_checkpoint(out/'best.pth');net.load_state_dict(ck['model_state_dict']);net.eval()
    final_best_epoch,final_best_val=ck['best_epoch'],ck['best_val_f1']
    del ck,optimizer,scheduler
    net.zero_grad(set_to_none=True)
    if device.type=='cuda':torch.cuda.empty_cache()
    old_key=read_json(marker).get('best_sha256') if marker.exists() else None
    best_hash=sha256(out/'best.pth')
    # 没有 complete 标记时，已保存结果仍可复用，但权重改变后必须重做评估。
    invalidate=trained or (old_key is not None and old_key!=best_hash)
    if invalidate or not (out/'classification_metrics.json').exists():
        evaluate(loaders['test'],net,device,criterion,'Test',out/'classification_metrics.json')
    if not a.no_profile_efficiency and (invalidate or not (out/'efficiency_metrics.json').exists()):
        write_json(profile(datasets['test'],loaders['test'],net,device,a),out/'efficiency_metrics.json')
    if state_required:
        for split in ('val','test'):
            target=out/'state_ablation'/split/'state_ablation.json'
            if invalidate or not target.exists():
                evaluate_state(loaders[split],net,device,datasets[split],target.parent,a.train_seed)
    write_json({'epochs':a.epochs,'best_epoch':final_best_epoch,'best_val_macro_f1':final_best_val,
                'best_sha256':best_hash,'profile_efficiency':not a.no_profile_efficiency},marker)
    print('COMPLETE:',out,flush=True)

if __name__=='__main__':run(parse_args())
