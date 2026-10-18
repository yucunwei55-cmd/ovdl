"""AF vs non-AF post-hoc evaluation; thresholds selected on validation ONLY."""
import math
import numpy as np
from sklearn.metrics import (confusion_matrix,roc_auc_score,average_precision_score,
                            roc_curve,precision_recall_curve,matthews_corrcoef)


def ratio(n,d):return float(n/d) if d else None


def wilson_interval(success,total,z=1.959963984540054):
    if total==0:return None
    p=success/total;den=1+z*z/total
    middle=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,middle-radius),min(1.,middle+radius)]


def binary_metrics(y,decision,score):
    y=np.asarray(y,dtype=bool);decision=np.asarray(decision,dtype=bool);score=np.asarray(score,dtype=float)
    if not (y.shape==decision.shape==score.shape) or not len(y):raise ValueError('Invalid binary metric arrays.')
    tn,fp,fn,tp=map(int,confusion_matrix(y,decision,labels=[False,True]).ravel())
    sensitivity,specificity=ratio(tp,tp+fn),ratio(tn,tn+fp)
    both=len(np.unique(y))==2
    return dict(tp=tp,tn=tn,fp=fp,fn=fn,af_records=tp+fn,non_af_records=tn+fp,
                missed_af_count=fn,missed_af_fraction=ratio(fn,tp+fn),
                sensitivity=sensitivity,recall=sensitivity,specificity=specificity,
                sensitivity_ci95=wilson_interval(tp,tp+fn),specificity_ci95=wilson_interval(tn,tn+fp),
                precision=ratio(tp,tp+fp),ppv=ratio(tp,tp+fp),npv=ratio(tn,tn+fn),
                false_positive_rate=ratio(fp,fp+tn),accuracy=ratio(tp+tn,len(y)),
                f1=ratio(2*tp,2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                balanced_accuracy=(sensitivity+specificity)/2 if both else None,
                mcc=float(matthews_corrcoef(y,decision)) if both else None,
                auc=float(roc_auc_score(y,score)) if both else None,
                average_precision=float(average_precision_score(y,score)) if both else None,
                brier_score=float(np.mean((score-y.astype(float))**2)))


def threshold_candidates(y,score):
    """Tie-aware candidates, O(N log N); no repeated O(N^2) confusion matrices."""
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    if not len(y) or len(np.unique(y))!=2:
        raise ValueError('Both AF/non-AF must exist in validation to select operating points.')
    if not np.isfinite(score).all() or ((score<0)|(score>1)).any():raise ValueError('Invalid AF probabilities.')
    order=np.argsort(-score,kind='stable');s=score[order];labels=y[order]
    last=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(labels)[last];fp=(last+1)-tp
    npos=int(y.sum());nneg=len(y)-npos
    thresholds=np.r_[np.nextafter(s[0],np.inf),s[last]]
    tp=np.r_[0,tp];fp=np.r_[0,fp];fn=npos-tp;tn=nneg-fp
    return dict(threshold=thresholds,tp=tp,fp=fp,fn=fn,tn=tn,
                sensitivity=tp/npos,specificity=tn/nneg,
                f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)!=0))


def choose_operating_points(val_y,val_score,sensitivity_targets=(.90,.95),specificity_targets=(.90,.95)):
    """Constraints are empirical validation targets, not clinical standards."""
    y=np.asarray(val_y,dtype=bool);score=np.asarray(val_score,dtype=float)
    table=threshold_candidates(y,score);result={}
    policies=[('validation_max_f1','f1',None)]
    policies += [(f'validation_sensitivity_{int(round(t*100))}','sensitivity',t) for t in sensitivity_targets]
    policies += [(f'validation_specificity_{int(round(t*100))}','specificity',t) for t in specificity_targets]
    for name,kind,target in policies:
        if target is None:
            candidates=np.arange(len(table['threshold']));objective='f1';secondary='specificity'
        else:
            candidates=np.flatnonzero(table[kind]>=target-1e-12)
            objective='specificity' if kind=='sensitivity' else 'sensitivity'
            secondary='f1'
        # Highest objective, then secondary score, then highest threshold.
        best=max(candidates.tolist(),key=lambda i:(table[objective][i],table[secondary][i],table['threshold'][i]))
        threshold=float(table['threshold'][best])
        result[name]=dict(threshold=threshold,selection_split='validation',
            policy='maximize validation F1' if target is None else f'maximize {objective} subject to {kind}>={target}',
            constraint_metric=None if target is None else kind,constraint_target=target,
            validation=binary_metrics(y,score>=threshold,score))
    return result


def af_evaluation(y,prob,operating_points):
    y=np.asarray(y);prob=np.asarray(prob);binary=y==1;score=prob[:,1]
    result=dict(score='softmax(mean time-step logits)[AF]',
                task='AF vs pooled SNR+OTHER; secondary evaluation of the three-class model, no binary retraining',
                argmax=binary_metrics(binary,prob.argmax(1)==1,score),
                threshold_0_5=binary_metrics(binary,score>=.5,score),operating_points={})
    for name,point in operating_points.items():
        test=binary_metrics(binary,score>=point['threshold'],score)
        metric,target=point['constraint_metric'],point['constraint_target']
        result['operating_points'][name]={**point,'test':test,
            'test_constraint_met':None if target is None or test[metric] is None else bool(test[metric]>=target),
            'note':'Constraint chosen on validation; test sensitivity/specificity may not meet the same target.'}
    return result


def curve_arrays(y,score):
    if len(np.unique(y))<2:return None,None
    fpr,tpr,threshold=roc_curve(y,score,drop_intermediate=False)
    precision,recall,pr_threshold=precision_recall_curve(y,score)
    return dict(fpr=fpr,tpr=tpr,threshold=threshold),dict(precision=precision,recall=recall,threshold=np.r_[pr_threshold,np.nan])
