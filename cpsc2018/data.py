"""CPSC2018 AF-priority mapping and corrected fixed/variable preprocessing."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy import signal
from preprocessing import preprocess_ecg
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,Sampler

CLASSES=['SNR','AF','OTHER']
LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
DIAGNOSES=['SNR','AF','IAVB','LBBB','RBBB','PAC','PVC','STD','STE']


def _record_key(value):
    # Match original IDs to labels.csv IDs such as Training_WFDB\\A0001.
    return Path(str(value).strip().replace('\\','/')).stem


def _diagnosis_matrix(frame):
    """Require a complete nine-column multi-hot table or all three diagnosis slots."""
    if all(c in frame for c in DIAGNOSES):
        values=frame[DIAGNOSES].apply(pd.to_numeric, errors='raise')
        if not values.isin([0,1]).all().all():
            raise ValueError('All nine diagnosis columns must contain 0/1 without missing values.')
        return values.astype(int).to_numpy(), 'complete nine-column multi-hot diagnoses'
    normalized={str(c).strip().lower().replace('_','').replace(' ',''):c for c in frame.columns}
    slot_names=['firstlabel','secondlabel','thirdlabel']
    if not all(c in normalized for c in slot_names):
        raise ValueError('AF-priority mapping needs all nine multi-hot diagnosis columns or '+
                         'First_label, Second_label and Third_label. First_label alone loses later AF diagnoses.')
    result=np.zeros((len(frame),9),dtype=int)
    for j,name in enumerate(slot_names):
        raw=frame[normalized[name]]
        clean=raw.replace(r'^\s*$',np.nan,regex=True)
        vals=pd.to_numeric(clean,errors='raise')
        if j>0:vals=vals.mask(vals==0)
        if j==0 and vals.isna().any():raise ValueError('Missing First_label diagnosis.')
        present=vals.notna()
        if not vals[present].isin(range(1,10)).all():
            raise ValueError('Diagnosis slots must be class codes 1..9 or empty later slots.')
        indices=np.flatnonzero(present.to_numpy())
        result[indices,vals[present].astype(int).to_numpy()-1]=1
    return result, 'all three original diagnosis slots'


def map_labels(df,expected_af_count=0,reference_csv=None):
    """AF in any position -> AF; pure SNR -> SNR; every remaining record -> OTHER."""
    df=df.copy()
    if 'patient_id' not in df:raise ValueError('Need patient_id in labels CSV.')
    if df.patient_id.isna().any():raise ValueError('Missing record IDs in labels CSV.')
    df['patient_id']=df.patient_id.astype(str)
    keys=df.patient_id.map(_record_key)
    if keys.duplicated().any():raise ValueError('Duplicate normalized record IDs in labels CSV.')
    try:
        diagnoses,source=_diagnosis_matrix(df)
    except ValueError:
        # Invalid complete multi-hot inputs must never be silently replaced.
        if all(c in df for c in DIAGNOSES) or reference_csv is None:raise
        ref=pd.read_csv(reference_csv)
        id_col=next((c for c in ['Recording','patient_id','recording','record_id'] if c in ref),None)
        if id_col is None or ref[id_col].isna().any():raise ValueError('Invalid reference record IDs.')
        ref_keys=ref[id_col].map(_record_key)
        if ref_keys.duplicated().any():raise ValueError('Duplicate reference record IDs.')
        matrix,_=_diagnosis_matrix(ref)
        lookup=pd.DataFrame(matrix,index=ref_keys,columns=DIAGNOSES)
        matched=lookup.reindex(keys.to_numpy())
        if matched.isna().any().any():raise ValueError('Missing record diagnoses in reference.')
        diagnoses=matched.to_numpy(dtype=int)
        source=str(Path(reference_csv).resolve())
    for i,c in enumerate(DIAGNOSES):df[c]=diagnoses[:,i]
    # A record with no diagnosis follows the requested remaining-record rule.
    af=diagnoses[:,1]==1
    abnormal=diagnoses[:,1:].any(axis=1)
    pure_normal=(diagnoses[:,0]==1)&~abnormal
    df['_y']=np.where(af,1,np.where(pure_normal,0,2))
    af_count=int(af.sum())
    if expected_af_count and af_count!=expected_af_count:
        raise ValueError(f'Any-diagnosis AF count={af_count}, expected {expected_af_count}. '+
                         'Inspect complete labels; use --expected-af-count 0 only for intentional subsets.')
    other=diagnoses[:,2:].any(axis=1)
    audit=dict(mapping='AF in any diagnosis -> AF; SNR with no abnormal diagnoses -> SNR; remaining -> OTHER',
               records=len(df),diagnosis_source=source,
               class_counts={c:int((df._y==i).sum()) for i,c in enumerate(CLASSES)},
               any_label_af_count=af_count,af_with_other_diagnoses=int((af&other).sum()),
               af_assigned_other=int((af&(df._y==2)).sum()),
               normal_with_abnormal_diagnoses=int(((diagnoses[:,0]==1)&abnormal).sum()),
               records_without_diagnoses=int((diagnoses.sum(axis=1)==0).sum()))
    return df,audit


def _find_reference(root,label_path,reference_csv=None):
    if reference_csv is not None:
        path=Path(reference_csv).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        return path
    candidates=[label_path.parent/'REFERENCE.csv',root/'REFERENCE.csv',
                root.parent/'REFERENCE.csv',root.parent.parent/'REFERENCE.csv',
                Path(__file__).resolve().parents[1]/'REFERENCE.csv',Path.cwd()/'REFERENCE.csv']
    return next((p for p in candidates if p.is_file()),None)


def make_splits(data_dir,label_csv,out_dir,split_seed=42,split_mode='folds',expected_af_count=1221,reference_csv=None):
    root=Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if label_csv:
        path=Path(label_csv).resolve()
    else:
        candidates=[root/'labels.csv',root.parent/'labels.csv']
        path=next((p for p in candidates if p.is_file()),candidates[0])
    raw=pd.read_csv(path,dtype={'patient_id':str})
    reference_path=None if (all(c in raw for c in DIAGNOSES) or all(any(str(c).strip().lower().replace('_','').replace(' ','')==name for c in raw) for name in ('firstlabel','secondlabel','thirdlabel'))) else _find_reference(root,path,reference_csv)
    df,audit=map_labels(raw,expected_af_count,reference_path)
    if split_mode=='folds':
        if 'fold' not in df:
            raise ValueError('Default split uses existing fold column; use --split-mode stratified if absent.')
        df['fold']=pd.to_numeric(df.fold,errors='raise').astype(int)
        if not set(df.fold).issubset(set(range(1,11))) or set(df.fold)!=set(range(1,11)):
            raise ValueError('fold must represent all 10 original folds, numbered 1..10.')
        order=np.random.RandomState(split_seed).permutation(np.arange(1,11))
        fold_sets=[order[:8],order[8:9],order[9:]]
        subsets=[df[df.fold.isin(f)].copy() for f in fold_sets]
        split_info={'mode':'existing 10 folds, permuted 8/1/1 as uploaded utils.split_data',
                    'folds':{n:f.tolist() for n,f in zip(['train','val','test'],fold_sets)}}
    else:
        ids=np.arange(len(df))
        tr,temp=train_test_split(ids,test_size=.2,stratify=df._y,random_state=split_seed)
        va,te=train_test_split(temp,test_size=.5,stratify=df._y.iloc[temp],random_state=split_seed)
        subsets=[df.iloc[i].copy() for i in [tr,va,te]]
        split_info={'mode':'record-level stratified 8/1/1'}
    out=Path(out_dir);(out/'splits').mkdir(parents=True,exist_ok=True)
    paths=[];split_counts={}
    for name,subset in zip(['train','val','test'],subsets):
        if set(subset._y)!=set(range(3)):
            raise ValueError(f'{name} lacks one of the three classes; cannot select/report all requested AF operating points.')
        p=out/'splits'/f'{name}.csv';subset.to_csv(p,index=False);paths.append(str(p))
        split_counts[name]={c:int((subset._y==i).sum()) for i,c in enumerate(CLASSES)}
    id_sets=[set(s.patient_id) for s in subsets]
    assert not any(id_sets[i]&id_sets[j] for i in range(3) for j in range(i))
    manifest=dict(split_seed=split_seed,label_csv=str(path),label_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  reference_csv=str(reference_path) if reference_path is not None else None,
                  reference_csv_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest() if reference_path is not None else None,
                  label_audit=audit,split=split_info,split_class_counts=split_counts,
                  protocol='record-disjoint; no patient IDs available, not claimed patient-independent',
                  sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return str(root),paths,manifest


def number_of_windows(n,length,stride,duration):
    if duration=='fixed' or n<=length:return 1
    return 1+(n-length+stride-1)//stride




class ECGDataset(Dataset):
    """500 Hz input; corrected preprocessing with optional legacy preprocessing ablation."""
    def __init__(self,phase,data_dir,label_csv,duration='variable',window_length=9000,
                 time_steps=3,seed=42,augment=True,leads='all',preprocessing='corrected',tail_mode='anchor'):
        self.phase,self.root,self.duration=phase,Path(data_dir),duration
        self.labels=pd.read_csv(label_csv,dtype={'patient_id':str})
        self.classes,self.n_classes=CLASSES,3
        self.seed,self.epoch,self.augment=seed,0,augment
        if preprocessing not in ('corrected','legacy') or tail_mode not in ('anchor','pad'):raise ValueError('Invalid preprocessing/tail mode')
        self.preprocessing,self.tail_mode=preprocessing,tail_mode
        self.timestep_length,self.repeat_times=window_length,time_steps
        self.stride=max(1,int(window_length*.8))
        self.selected_leads=LEADS if leads=='all' else leads.split(',')
        if not self.selected_leads or len(set(self.selected_leads))!=len(self.selected_leads) or set(self.selected_leads)-set(LEADS):
            raise ValueError('leads must be all or comma-separated standard lead names, e.g. I,II,V1.')
        self.nleads=len(self.selected_leads)
        self.record_paths=[];self.lengths=[];self.channel_indices=[];self.window_counts=[]
        basename_index=None
        for pid in self.labels.patient_id:
            rel=str(pid).replace('\\','/');relative=Path(rel)
            if relative.suffix in ('.mat','.hea','.dat'):relative=relative.with_suffix('')
            candidates=[self.root/relative,self.root/relative.name]
            path=next((p for p in candidates if p.with_suffix('.hea').is_file()),None)
            if path is None:
                if basename_index is None:
                    basename_index={}
                    for hea in self.root.rglob('*.hea'):
                        basename_index.setdefault(hea.stem,[]).append(hea.with_suffix(''))
                found=basename_index.get(relative.name,[])
                if len(found)!=1:raise FileNotFoundError(f'No unique .hea for record {pid} under {self.root}.')
                path=found[0]
            h=wfdb.rdheader(str(path))
            if abs(h.fs-500.)>1e-6:raise ValueError(f'{pid}: sampling rate {h.fs}, expected 500 Hz; resample explicitly first.')
            if h.sig_len<=0:raise ValueError(f'Empty ECG {pid}')
            if len(set(h.sig_name))!=len(h.sig_name) or set(self.selected_leads)-set(h.sig_name):
                raise ValueError(f'{pid}: selected leads missing or duplicated in header {h.sig_name}.')
            self.record_paths.append(str(path));self.lengths.append(h.sig_len)
            self.channel_indices.append([h.sig_name.index(c) for c in self.selected_leads])
            self.window_counts.append(number_of_windows(h.sig_len,window_length,self.stride,duration))
        # Resource helper expects base windows, multiplied exactly once by repeat_times.
        self.time_steps_list=self.window_counts
        self.actual_time_steps_list=[n*time_steps for n in self.window_counts]

    def __len__(self):return len(self.labels)

    def __getitem__(self,index):
        path=self.record_paths[index]
        try:x,fields=wfdb.rdsamp(path)
        except Exception as exc:raise RuntimeError(f'Cannot load {path}; no synthetic zero record substituted.') from exc
        if len(x)!=self.lengths[index] or not np.isfinite(x).all():
            raise ValueError(f'Changed length or invalid signal {path}')
        x=preprocess_ecg(x[:,self.channel_indices[index]],self.duration,self.timestep_length,
                         self.preprocessing,self.phase=='train' and self.augment,
                         (self.seed+self.epoch*100003+index*997)%(2**32))
        chunks=[]
        for j in range(self.window_counts[index]):
            start=0 if self.duration=='fixed' else j*self.stride
            if self.duration=='variable' and self.tail_mode=='anchor' and len(x)>self.timestep_length:
                start=min(start,len(x)-self.timestep_length)
            chunk=x[start:start+self.timestep_length]
            if len(chunk)<self.timestep_length:chunk=np.pad(chunk,((0,self.timestep_length-len(chunk)),(0,0)))
            chunks.append(chunk)
        sequence=np.repeat(np.stack(chunks).astype(np.float32),self.repeat_times,axis=0)
        assert len(sequence)==self.actual_time_steps_list[index]
        return torch.from_numpy(sequence.copy()),torch.tensor(int(self.labels.iloc[index]._y),dtype=torch.long)


class TimeStepGroupSampler(Sampler):
    def __init__(self,time_steps_list,batch_size,seed=42,shuffle=True):
        self.batch_size,self.seed,self.shuffle,self.epoch=batch_size,seed,shuffle,0
        groups={}
        for i,t in enumerate(time_steps_list):groups.setdefault(int(t),[]).append(i)
        self.group_indices=list(groups.values())
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);order=np.arange(len(self.group_indices))
        if self.shuffle:rng.shuffle(order)
        for k in order:
            indices=np.array(self.group_indices[k])
            if self.shuffle:rng.shuffle(indices)
            for i in range(0,len(indices),self.batch_size):yield indices[i:i+self.batch_size].tolist()
    def __len__(self):return sum((len(g)+self.batch_size-1)//self.batch_size for g in self.group_indices)


def collate(batch):
    x,y=zip(*batch)
    if len({tuple(v.shape) for v in x})!=1:raise ValueError('Exact-T batch has mixed shapes.')
    return torch.stack(x),torch.stack(y)
