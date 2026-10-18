"""统计完成实验的均值和样本标准差；缺失种子不填0。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from io_utils import read_json,write_json

def summarize(root):
    root=Path(root);rows=[]
    for marker in sorted(root.glob('seed*/*/complete.json')):
        out=marker.parent;meta=read_json(out/'run_config.json');c=meta['config']
        test=read_json(out/'classification_metrics.json')['metrics'];history=read_json(out/'training_history.json')
        ef=read_json(out/'efficiency_metrics.json') if (out/'efficiency_metrics.json').exists() else {}
        profile=ef.get('grouped_inference',{})
        # 兼容导入 V2 的原始 profile 命名，不重新测量旧实测结果。
        if not profile:
            profile=ef.get('grouped_benchmark',ef.get('grouped',ef.get('exact_t_grouped',{})))
        potential=sum(x['potential_optimizer_steps'] for x in history);updates=sum(x['optimizer_steps'] for x in history)
        row={'duration':c['duration'],'variant':c['variant'],'train_seed':c['train_seed'],'split_seed':c['split_seed'],
             'epochs':len(history),'best_epoch':read_json(marker)['best_epoch'],'imported_v2':bool(meta.get('imported_v2')),
             'accuracy':test['accuracy'],'f1_macro':test['f1_macro'],'f1_weighted':test['f1_weighted'],
             'f1_NAO':test['f1_NAO'],'auc_ovr_macro':test.get('auc_ovr_macro'),
             'af_f1':test['af_vs_non_af'].get('binary_f1'),
             'af_sensitivity':test['af_vs_non_af']['sensitivity'],'af_specificity':test['af_vs_non_af']['specificity'],
             'af_auc':test['af_vs_non_af']['auc'],'potential_optimizer_steps':potential,'optimizer_steps':updates,
             'update_skip_fraction':1-updates/potential,'training_wall_s':sum(x['epoch_wall_time_s'] for x in history),
             'historical_trace_cache_peak_bytes':max(x['historical_trace_cache_peak_bytes'] for x in history),
             'full_resolution_trace_equivalent_peak_bytes':max(x['full_resolution_trace_equivalent_peak_bytes'] for x in history),
             'training_peak_cuda_memory_mb':max((x['peak_cuda_memory_allocated_mb'] for x in history if x.get('peak_cuda_memory_allocated_mb') is not None),default=None),
             'latency_ms_per_record':profile.get('latency_ms_per_record',profile.get('latency_ms_per_record_model_only',profile.get('model_latency_ms_per_record')))}
        energies=[x.get('gpu_energy_j') for x in history]
        row['training_gpu_energy_j']=sum(energies) if energies and all(x is not None for x in energies) else None
        row['training_seconds_per_epoch'] = row['training_wall_s'] / len(history)
        row['optimizer_steps_per_epoch'] = updates / len(history)
        row['inference_peak_allocated_mib'] = profile.get('peak_cuda_memory_allocated_mb')
        row['dense_gmac_per_record'] = ef.get('operations', {}).get('conv_linear_macs_per_record_mean')
        if row['dense_gmac_per_record'] is not None: row['dense_gmac_per_record'] /= 1e9
        row['firing_fraction'] = ef.get('overall_lif_firing_fraction')
        rows.append(row)
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(root/'summary.csv',index=False)
    numeric=[k for k in df.select_dtypes(include='number').columns if k not in ('train_seed','split_seed')]
    stats=[]
    for (duration,variant),group in df.groupby(['duration','variant'],sort=True):
        item={'duration':duration,'variant':variant,'n_seeds':len(group),'train_seeds':','.join(map(str,group.train_seed))}
        item['split_seeds']=','.join(map(str,group.split_seed))
        item['seed_pairs']=','.join(f'{t}:{s}' for t,s in zip(group.train_seed,group.split_seed))
        for key in numeric:
            values=group[key].dropna();item[key+'_n']=len(values)
            item[key+'_mean']=float(values.mean()) if len(values) else None
            item[key+'_std']=float(values.std(ddof=1)) if len(values)>1 else None
        stats.append(item)
    pd.DataFrame(stats).to_csv(root/'summary_mean_std.csv',index=False)
    # 同一训练/划分种子的主方法与消融做配对，不把两种时长当作相同输入。
    pairs=[]
    for duration in df.duration.unique():
        local=df[df.duration==duration];base=local[local.variant=='main']
        for variant in local.variant.unique():
            if variant=='main':continue
            matched=base.merge(local[local.variant==variant],on=['train_seed','split_seed'],suffixes=('_main','_ablation'))
            if matched.empty:continue
            delta=matched.f1_macro_main-matched.f1_macro_ablation
            pairs.append({'duration':duration,'ablation':variant,'n_pairs':len(delta),
                          'f1_main_minus_ablation_mean':float(delta.mean()),
                          'f1_main_minus_ablation_std':float(delta.std(ddof=1)) if len(delta)>1 else None})
    write_json({'paired_differences':pairs,'note':'std uses ddof=1; paired train/split seeds; variation includes both training and partition randomness'},root/'paired_summary.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    summarize(parser.parse_args().root)
