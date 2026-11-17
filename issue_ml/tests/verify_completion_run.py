"""对已完成实验检查标签、时间隔离和保存模型；不重新训练。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from issue_ml.completion.metrics import evaluate

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args()
    output=Path(args.output);config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    prepared=joblib.load(output/'prepared.joblib');bundle=joblib.load(output/'selected_model.joblib')
    selected=json.loads((output/'selection.json').read_text(encoding='utf-8'))
    audit=json.loads((output/'audit.json').read_text(encoding='utf-8'))
    for source,key in [('issues','issue_sha256'),('comments','comment_sha256')]:
        assert hashlib.sha256(Path(config[source]).read_bytes()).hexdigest()==audit[key]
    for (h,d),frame in prepared['frames'].items():
        assert frame.age_days.ge(d).all()
        assert (~(frame.observed_closed & frame.close_hours.le(h))).all()
        expected=(frame.observed_closed & frame.state_reason.eq('completed') & frame.close_hours.gt(h) & frame.close_hours.le(d*24)).astype(int)
        assert np.array_equal(frame.y,expected)
        train=frame[frame.split.eq('train')];val=frame[frame.split.eq('validation')];test=frame[frame.split.eq('test')]
        assert (train.created_at+pd.Timedelta(days=d)).max()<val.created_at.min()
        assert (val.created_at+pd.Timedelta(days=d)).max()<test.created_at.min()
        assert set(train.number).isdisjoint(val.number) and set(train.number).isdisjoint(test.number) and set(val.number).isdisjoint(test.number)
    frame=prepared['frames'][(selected['h'],selected['d'])]
    test=frame[frame.split.eq('test')].sort_values('created_at')
    scores=bundle['forest'].predict_proba(bundle['features'].transform(test))[:,1]
    saved=pd.read_csv(output/'selected_predictions.csv')
    assert test.number.tolist()==saved.number.tolist()
    assert np.allclose(saved.p_completed,scores,rtol=0,atol=1e-12)
    result=evaluate(test.y,scores,bundle['threshold'])
    metrics=pd.read_csv(output/'test_metrics.csv')
    row=metrics[(metrics.h==selected['h'])&(metrics.d==selected['d'])&(metrics.variant==selected['variant'])&metrics.model.eq('random_forest')].iloc[0]
    for key in ['roc_auc','average_precision','brier','precision','recall','f1','top20_incomplete_rate']:
        assert abs(result[key]-row[key])<1e-12
    proof={'all_12_cohort_labels_checked':True,'all_temporal_maturity_gaps_checked':True,
           'raw_data_hashes_match':True,'selected_saved_model_predictions_match':True,
           'selected_test_rows':len(test),'metrics_recomputed':True}
    (output/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
