import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from .data import prepare, VARIANTS, TEXT_VARIANTS, TIME, STRUCTURE, EARLY, PROXY
from .metrics import evaluate, select_threshold, paired_bootstrap

class CompletionFeatures:
    """随机森林的特征变换器。

    结构化列由 VARIANTS[variant] 给出；若 variant 在 TEXT_VARIANTS 中，
    还会在训练集上拟合 TF-IDF→TruncatedSVD 文本 pipeline 并追加到特征矩阵。

    变体语义：
    - early_only      : 时间 + 早期互动计数，无文本、无结构、无身份代理。
    - text_structure  : 时间 + 结构 + 文本 SVD，无早期计数。
    - structure_early : 时间 + 结构 + 早期计数 + 身份代理，无文本。
    - combined        : 时间 + 结构 + 早期计数 + 身份代理 + 文本 SVD。
    （combined 和 structure_early 的结构化列相同，区别仅在于是否拼接文本 SVD。）
    """
    def __init__(self, variant, config):
        if variant not in VARIANTS:
            raise ValueError(f'未知特征变体 {variant!r}，合法值：{sorted(VARIANTS)}')
        self.variant = variant
        self.columns = VARIANTS[variant]
        self.config  = config
        self.text_pipeline = None

    def fit(self, frame):
        """在训练集上拟合文本 pipeline（如适用）。"""
        if self.variant in TEXT_VARIANTS:
            t = self.config['text']
            self.text_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 2), min_df=t['min_df'], max_df=0.95,
                    max_features=t['max_features'], sublinear_tf=True,
                )),
                ('svd', TruncatedSVD(
                    n_components=t['svd_components'], random_state=self.config['seed'],
                )),
            ])
            self.text_pipeline.fit(frame.text)
        return self

    def transform(self, frame):
        """将 frame 转换为特征矩阵（float32）。"""
        arrays = [frame[self.columns].to_numpy(dtype=np.float32)]
        if self.text_pipeline is not None:
            arrays.append(self.text_pipeline.transform(frame.text).astype(np.float32))
        result = np.hstack(arrays)
        if not np.isfinite(result).all():
            raise ValueError('特征矩阵中存在非有限值（NaN 或 Inf），请检查输入数据')
        return result

    def names(self):
        """返回所有特征列名（结构化列 + 可选的 text_svd_* 列）。"""
        text_names = (
            [f'text_svd_{i}' for i in range(self.config['text']['svd_components'])]
            if self.text_pipeline else []
        )
        return self.columns + text_names

def load_config(path):
    path=Path(path).resolve(); config=json.loads(path.read_text(encoding='utf-8-sig'))
    root=(path.parent/config.get('project_root','..')).resolve()
    for k in ['issues','comments','output']:config[k]=str((root/config[k]).resolve())
    return config

def tune(frame,variant,config,h,d,output):
    train=frame[frame.split.eq('train')].sort_values('created_at')
    validation=frame[frame.split.eq('validation')].sort_values('created_at')
    if train.y.nunique()!=2 or validation.y.nunique()!=2:raise ValueError('Missing class in train or validation')
    features=CompletionFeatures(variant,config).fit(train)
    x=features.transform(train);v=features.transform(validation)
    rows=[];best=None
    for i,params in enumerate(config['forest_candidates']):
        model=RandomForestClassifier(**params,random_state=config['seed'],n_jobs=config['n_jobs'])
        model.fit(x,train.y)
        p=model.predict_proba(v)[:,1]
        threshold=select_threshold(validation.y,p)
        score=evaluate(validation.y,p,threshold)
        rows.append({'h':h,'d':d,'variant':variant,'candidate':i,'params':json.dumps(params),**score})
        # 参数选择优先 AP，AP 相同时选择概率误差较小的模型。
        rank=(score['average_precision'],-score['brier'])
        if best is None or rank>best[0]:best=(rank,model,threshold,score,i,p)
    print(f'VALIDATION h={h}, d={d}, {variant}: AP={best[3]["average_precision"]:.4f}, AUC={best[3]["roc_auc"]:.4f}, candidate={best[4]}',flush=True)
    bundle={'features':features,'forest':best[1],'threshold':best[2],'validation_metrics':best[3],
            'h':h,'d':d,'variant':variant,'candidate':best[4],'config':config,'train_prior':float(train.y.mean())}
    joblib.dump(bundle,output/'models'/f'h{h}_d{d}_{variant}.joblib')
    return bundle,rows

def window_score(bundle):
    m=bundle['validation_metrics']
    lift=m['average_precision']/m['positive_rate']
    # 提前性折扣：同等AP提升时，等待更少、剩余行动时间更多者优先。
    return lift*(1-bundle['h']/(24*bundle['d']))

def run(config,audit_only=False):
    output=Path(config['output'])
    cache=output/'prepared.joblib'
    if not output.exists():output.mkdir(parents=True)
    saved=output/'config.json'
    if saved.exists() and json.loads(saved.read_text(encoding='utf-8'))!=config:
        raise ValueError('Configuration differs from existing run; use a new output directory')
    saved.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    if cache.exists():
        prepared=joblib.load(cache);frames,audit,flow=prepared['frames'],prepared['audit'],prepared['flow']
    else:
        frames,audit,flow=prepare(config,output)
        joblib.dump({'frames':frames,'audit':audit,'flow':flow},cache,compress=3)
    if audit_only:return
    if (output/'selection.json').exists():raise FileExistsError('Experiment already selected; choose a new output directory')
    (output/'models').mkdir(exist_ok=True)
    bundles={};grid=[]
    def get(h,d,variant):
        key=(h,d,variant)
        if key not in bundles:
            bundle,rows=tune(frames[(h,d)],variant,config,h,d,output)
            bundles[key]=bundle;grid.extend(rows)
            pd.DataFrame(grid).to_csv(output/'tuning_validation.csv',index=False,encoding='utf-8-sig')
        return bundles[key]
    for h in config['hours']:get(h,30,'structure_early')
    # 在最优及时性调整AP提升的95%范围内，选择最早观察时点。
    best_utility=max(window_score(get(h,30,'structure_early')) for h in config['hours'])
    selected_h=min(h for h in config['hours'] if window_score(get(h,30,'structure_early'))>=.95*best_utility)
    for d in config['days']:get(selected_h,d,'structure_early')
    selected_d=max(config['days'],key=lambda d:window_score(get(selected_h,d,'structure_early')))
    for variant in VARIANTS:get(selected_h,selected_d,variant)
    # 额外在固定30天任务比较同样的四种特征，避免改期限后无法解释原始30天目标。
    for variant in VARIANTS:get(selected_h,30,variant)
    selected_variant=max(VARIANTS,key=lambda v:get(selected_h,selected_d,v)['validation_metrics']['average_precision'])
    selection={'h':selected_h,'d':selected_d,'variant':selected_variant,
               'selection_rule':'validation AP per parameter; window utility=(AP/positive_rate)*(1-h/(24*d)); choose earliest h within 95% of best utility; d maximizes utility; variant maximizes validation AP',
               'test_used_for_selection':False,'threshold_rule':'maximum validation F1; no test tuning',
               'train_refit_after_validation':False,'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__}}
    # 先持久化选择，后开始测试集转换和预测。
    (output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SELECTION LOCKED',selection,flush=True)
    results=[];predictions=[];membership=[];summary=[]
    for (h,d,variant),bundle in bundles.items():
        frame=frames[(h,d)]
        for split in ['train','validation','test']:
            sub=frame[frame.split.eq(split)]
            summary.append({'h':h,'d':d,'variant':variant,'split':split,'n':len(sub),'positive':int(sub.y.sum()),'rate':float(sub.y.mean())})
        test=frame[frame.split.eq('test')].sort_values('created_at')
        p=bundle['forest'].predict_proba(bundle['features'].transform(test))[:,1]
        model_metrics=evaluate(test.y,p,bundle['threshold'])
        prior=np.full(len(test),bundle['train_prior'])
        baseline=evaluate(test.y,prior,.5)
        # 常数排序没有信息；队列期望值就是总体未完成率，不解读同分次序。
        baseline['top20_incomplete_rate']=baseline['overall_incomplete_rate'];baseline['top20_lift']=1.
        for label,metrics in [('random_forest',model_metrics),('train_prior',baseline)]:
            results.append({'h':h,'d':d,'variant':variant,'model':label,**metrics})
        preds=test[['number','created_at','title','y','comment_count_matches']].copy()
        preds['p_completed']=p;preds['prediction']=(p>=bundle['threshold']).astype(int)
        preds['h']=h;preds['d']=d;preds['variant']=variant
        predictions.append(preds)
        columns=['number','created_at','split','y','close_hours','state_reason']
        member=frame[columns].copy();member['h']=h;member['d']=d;membership.append(member)
    result_table=pd.DataFrame(results);pred_table=pd.concat(predictions)
    result_table.to_csv(output/'test_metrics.csv',index=False,encoding='utf-8-sig')
    pred_table.to_csv(output/'test_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(output/'split_counts.csv',index=False,encoding='utf-8-sig')
    pd.concat(membership).drop_duplicates(['number','h','d']).to_csv(output/'sample_membership.csv',index=False,encoding='utf-8-sig')
    validation_rows=[{'h':h,'d':d,'variant':v,'candidate':b['candidate'],'utility':window_score(b),**b['validation_metrics']} for (h,d,v),b in bundles.items()]
    pd.DataFrame(validation_rows).to_csv(output/'window_validation.csv',index=False,encoding='utf-8-sig')
    selected=bundles[(selected_h,selected_d,selected_variant)]
    joblib.dump(selected,output/'selected_model.joblib')
    sf=frames[(selected_h,selected_d)];st=sf[sf.split.eq('test')].sort_values('created_at')
    chosen=pred_table[(pred_table.h==selected_h)&(pred_table.d==selected_d)&(pred_table.variant==selected_variant)]
    chosen.to_csv(output/'selected_predictions.csv',index=False,encoding='utf-8-sig')
    chosen[chosen.y!=chosen.prediction].to_csv(output/'selected_errors.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'feature':selected['features'].names(),'importance':selected['forest'].feature_importances_}).sort_values('importance',ascending=False).to_csv(output/'feature_importance.csv',index=False,encoding='utf-8-sig')
    # 同队列比较全部特征与文本结构，配对自助法仅反映该测试集不确定性。
    text_pred=pred_table[(pred_table.h==selected_h)&(pred_table.d==selected_d)&(pred_table.variant=='text_structure')].set_index('number')
    combined=pred_table[(pred_table.h==selected_h)&(pred_table.d==selected_d)&(pred_table.variant=='combined')].set_index('number').loc[text_pred.index]
    uncertainty=paired_bootstrap(text_pred.y,combined.p_completed,text_pred.p_completed,config['seed'])
    (output/'paired_uncertainty.json').write_text(json.dumps(uncertainty,indent=2),encoding='utf-8')
    sensitivity=evaluate(chosen.loc[chosen.comment_count_matches,'y'],chosen.loc[chosen.comment_count_matches,'p_completed'],selected['threshold'])
    (output/'comment_completeness_sensitivity.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf-8')
    from .report import write_report
    write_report(output,selection,audit,flow,result_table,pred_table)
    from .diagnostics import write_diagnostics
    write_diagnostics(output)
    print(result_table[result_table.model.eq('random_forest')][['h','d','variant','n','positive_rate','roc_auc','average_precision','brier','f1']].to_string(index=False),flush=True)
