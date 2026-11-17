import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve)

def select_threshold(y,p):
    precision, recall, thresholds = precision_recall_curve(y,p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[np.argmax(f1)])

def evaluate(y,p,threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if not len(y): return {'n':0}
    pred=p>=threshold
    count=max(1,int(np.ceil(.2*len(y))))
    # 稳定排序，恰好取前20%，不以分位阈值把所有同分记录都纳入。
    low=np.argsort(p,kind='stable')[:count]
    incomplete=1-y.mean()
    worst=float(1-y[low].mean())
    return {'n':len(y),'positives':int(y.sum()),'positive_rate':float(y.mean()),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'average_precision':float(average_precision_score(y,p)) if y.sum() else 0.,
            'brier':float(brier_score_loss(y,p)),'threshold':float(threshold),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'top20_count':count,'top20_incomplete_rate':worst,'overall_incomplete_rate':float(incomplete),
            'top20_lift':float(worst/incomplete) if incomplete else None,
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def paired_bootstrap(y,p,q,seed=42,repeats=500):
    y,p,q=map(np.asarray,(y,p,q));rng=np.random.default_rng(seed);aucs=[];aps=[];delta_auc=[];delta_ap=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y)); z=y[idx]
        if len(np.unique(z))<2:continue
        au=roc_auc_score(z,p[idx]);ap=average_precision_score(z,p[idx])
        aucs.append(au);aps.append(ap)
        delta_auc.append(au-roc_auc_score(z,q[idx]));delta_ap.append(ap-average_precision_score(z,q[idx]))
    return {k:np.quantile(v,[.025,.975]).tolist() for k,v in [('auc_ci95',aucs),('ap_ci95',aps),('delta_auc_ci95',delta_auc),('delta_ap_ci95',delta_ap)]}
