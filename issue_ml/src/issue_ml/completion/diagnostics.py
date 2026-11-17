"""实验后诊断，不改变已锁定的模型或阈值。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import paired_bootstrap

def write_diagnostics(output):
    output=Path(output)
    selected=pd.read_csv(output/'selected_predictions.csv')
    predictions=pd.read_csv(output/'test_predictions.csv')
    selection=json.loads((output/'selection.json').read_text(encoding='utf-8'))
    result={'selected_vs_constant':paired_bootstrap(selected.y,selected.p_completed,np.full(len(selected),.5),repeats=500)}
    text=predictions[(predictions.h==selection['h']) & predictions.d.eq(30) & predictions.variant.eq('text_structure')].set_index('number')
    for variant in ['early_only','combined']:
        other=predictions[(predictions.h==selection['h']) & predictions.d.eq(30) & predictions.variant.eq(variant)].set_index('number').loc[text.index]
        result['30d_'+variant+'_vs_text_structure']=paired_bootstrap(text.y,other.p_completed,text.p_completed,repeats=500)
    result['selected_top_high_scores']={str(k):{'completed':int(selected.nlargest(k,'p_completed').y.sum()),'n':k} for k in [3,10,20,50,100]}
    (output/'additional_diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
