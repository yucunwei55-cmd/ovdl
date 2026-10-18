"""原子保存与固定划分；不覆盖数据根目录中的旧 CSV。"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)
def save_checkpoint(obj,path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp);os.replace(tmp,path)
def load_checkpoint(path,device='cpu'):
    # 只加载自己生成、可信的 checkpoint；RNG/优化器需 weights_only=False。
    try: return torch.load(path,map_location=device,weights_only=False)
    except TypeError: return torch.load(path,map_location=device)
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def make_splits(data_root,seed,out):
    ref=Path(data_root)/'training2017/REFERENCE.csv'
    df=pd.read_csv(ref,header=None,names=['patient_id','label'])
    if df.patient_id.duplicated().any() or df.label.isna().any() or not set(df.label)<=set(['N','A','O','~']):
        raise ValueError('REFERENCE.csv 必须为无重复记录的 N/A/O/~ 两列表')
    labels=df.label.astype(str).values
    train,temp=train_test_split(np.arange(len(df)),test_size=.2,stratify=labels,random_state=seed,shuffle=True)
    val,test=train_test_split(temp,test_size=.5,stratify=labels[temp],random_state=seed,shuffle=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,ids in zip(('train','val','test'),(train,val,test)):
        target=out/(name+'.csv');content=df.iloc[ids].to_csv(index=False,header=False)
        if target.exists() and target.read_text()!=content:
            raise ValueError('划分文件已有不同内容，拒绝覆盖: '+str(target))
        if not target.exists(): target.write_text(content)
        paths[name]=str(target)
    print(f'Split records: total={len(df)}, train={len(train)}, val={len(val)}, test={len(test)}',flush=True)
    return paths,{k:sha256(v) for k,v in paths.items()},sha256(ref)
