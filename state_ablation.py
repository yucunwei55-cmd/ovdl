"""同权重推理干预：每个新窗口开始时重置全部 LIF，而不改变分类权重。

历史梯度迹只在训练启用，因此该检查不能证明 q 的训练贡献。
指标覆盖整个验证/测试集，不再输出 single_window/multi_window 分组。
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from metrics import compute_classification_metrics
from engine import sequence_logits
from io_utils import write_json,read_json,load_checkpoint,sha256

def scores(y, pred):
    cm = np.bincount(4 * y + pred, minlength=16).reshape(4, 4)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2.0 * cm.diagonal(), denominator,
                   out=np.zeros(4), where=denominator != 0)
    af_total = cm[1].sum()
    return np.array([(pred == y).mean(), f1.mean(), f1[:3].mean(),
                     cm[1, 1] / af_total if af_total else np.nan])

def paired_summary(y, carry, reset, probabilities, bootstrap, seed):
    names = ['accuracy', 'f1_macro', 'f1_NAO', 'af_sensitivity']
    difference = scores(y, reset) - scores(y, carry)
    intervals = {name: None for name in names}
    if bootstrap:
        rng = np.random.default_rng(seed)
        strata = [np.flatnonzero(y == k) for k in range(4)]
        samples = []
        for _ in range(bootstrap):
            # Identical resampled records for both predictions; fixed class counts.
            ids = np.concatenate([rng.choice(s, len(s), replace=True)
                                  for s in strata if len(s)])
            samples.append(scores(y[ids], reset[ids]) - scores(y[ids], carry[ids]))
        samples = np.array(samples)
        for column, name in enumerate(names):
            if np.isfinite(samples[:, column]).all():
                intervals[name] = np.quantile(samples[:, column], [.025, .975]).tolist()
    lost = int(np.sum((carry == y) & (reset != y)))
    gained = int(np.sum((carry != y) & (reset == y)))
    return {
        'records': len(y), 'class_support': np.bincount(y, minlength=4).tolist(),
        'carry': compute_classification_metrics(y, carry, probabilities[0], num_classes=4),
        'reset_neurons': compute_classification_metrics(y, reset, probabilities[1], num_classes=4),
        'delta_reset_minus_carry': {k: float(v) if np.isfinite(v) else None
                                   for k, v in zip(names, difference)},
        'paired_stratified_bootstrap_95ci': intervals,
        'prediction_changes': int(np.sum(carry != reset)),
        'max_probability_difference':float(np.abs(probabilities[0]-probabilities[1]).max()),
        'mean_probability_difference':float(np.abs(probabilities[0]-probabilities[1]).mean()),
        'carry_correct_reset_wrong': lost, 'carry_wrong_reset_correct': gained,
        'exact_mcnemar_p_accuracy': float(binomtest(lost, lost + gained, .5).pvalue)
                                    if lost + gained else 1.0,
    }

@torch.no_grad()
def evaluate_state(loader,net,device,dataset,out,seed,bootstrap=2000):
    net.eval();ys=[];probs=[[],[]]
    for data,target in loader:
        data=data.to(device);ys.append(target.numpy())
        for index,reset in enumerate((False,True)):
            probs[index].append(sequence_logits(net,data,reset).softmax(1).cpu().numpy())
    y=np.concatenate(ys);prob=[np.concatenate(x) for x in probs];pred=[x.argmax(1) for x in prob]
    result=paired_summary(y,pred[0],pred[1],prob,bootstrap,seed)
    result['scope']='all records; paired same-weight inference intervention; all LIF reset at each window boundary'
    write_json(result,Path(out)/'state_ablation.json')
    # 顺序与精确 T sampler 相同，保留每条记录的窗数供离线分析，不输出分组指标。
    order=[i for batch in loader.batch_sampler for i in batch]
    rows={'record_id':dataset.labels.iloc[order].patient_id.values,'label':y,
          'ecg_windows':np.asarray(dataset.time_steps_list)[order],
          'carry_prediction':pred[0],'reset_prediction':pred[1]}
    for c,label in enumerate(('N','A','O','noise')):
        rows['carry_p_'+label]=prob[0][:,c];rows['reset_p_'+label]=prob[1][:,c]
    pd.DataFrame(rows).to_csv(Path(out)/'paired_predictions.csv',index=False)
    print('Same-weight state reset:',{k:result[k] for k in ('prediction_changes','delta_reset_minus_carry','max_probability_difference')},flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--data-dir')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--split',choices=('val','test'),default='test')
    args=parser.parse_args()
    if args.bootstrap<0:parser.error('bootstrap must be nonnegative')
    from train import build_loaders
    from model import from_config
    from sampling import seed_everything
    from types import SimpleNamespace
    run=Path(args.run_dir).resolve();meta=read_json(run/'run_config.json');c=meta['config']
    if args.data_dir:c['data_dir']=str(Path(args.data_dir).resolve())
    c['device']=args.device;a=SimpleNamespace(**c)
    if a.torch_threads:torch.set_num_threads(a.torch_threads)
    seed_everything(a.train_seed)
    checkpoint=load_checkpoint(run/'best.pth')
    if checkpoint['split_sha256']!=meta['split_sha256']:raise ValueError('Checkpoint split mismatch')
    for split in ('train','val','test'):
        if sha256(run/'splits'/(split+'.csv'))!=meta['split_sha256'][split]:raise ValueError('Split hash mismatch')
    datasets,loaders,generator,channels=build_loaders(a)
    net=from_config(c,channels,torch.device(args.device));net.load_state_dict(checkpoint['model_state_dict'])
    evaluate_state(loaders[args.split],net,torch.device(args.device),datasets[args.split],
                   run/'state_ablation'/args.split,a.train_seed,args.bootstrap)

if __name__=='__main__':main()
