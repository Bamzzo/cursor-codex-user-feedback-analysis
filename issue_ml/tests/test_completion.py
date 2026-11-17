import json
import sys
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from issue_ml.completion.data import assign_split, deduplicate, prepare, VARIANTS, TEXT_VARIANTS, STRUCTURE, PROXY, TIME, EARLY
from issue_ml.completion.metrics import evaluate, select_threshold

class CompletionContractTests(unittest.TestCase):
    def test_label_maturity_gap(self):
        config={'train_end':'2026-05-01T00:00:00Z','validation_start':'2026-05-02T00:00:00Z',
                'validation_end':'2026-06-01T00:00:00Z','test_start':'2026-07-01T00:00:00Z','test_end':'2026-08-01T00:00:00Z'}
        frame=pd.DataFrame({'created_at':pd.to_datetime(['2026-04-01'],utc=True)})
        with self.assertRaises(ValueError):assign_split(frame,config,30)

    def test_strict_early_features_exclude_snapshot_text_and_roles(self):
        self.assertFalse(set(VARIANTS['early_only']) & set(STRUCTURE+PROXY))
        forbidden={'comments','state','state_reason','closed_at','close_hours','age_days','labels','updated_at','reactions'}
        for cols in VARIANTS.values():self.assertFalse(set(cols)&forbidden)

    def test_combined_adds_text_on_top_of_structure_early(self):
        """combined 与 structure_early 结构化列相同，区别仅在于 TEXT_VARIANTS 控制文本 pipeline。"""
        self.assertEqual(VARIANTS['combined'], VARIANTS['structure_early'])
        self.assertIn('combined', TEXT_VARIANTS)
        self.assertNotIn('structure_early', TEXT_VARIANTS)
        # text_structure 也加文本 pipeline，但列组合不同
        self.assertIn('text_structure', TEXT_VARIANTS)
        self.assertNotEqual(VARIANTS['text_structure'], VARIANTS['combined'])

    def test_variant_columns_are_subsets_of_known_groups(self):
        """每个变体的列均由 TIME / STRUCTURE / EARLY / PROXY 组成，无未知列。"""
        all_known = set(TIME + STRUCTURE + EARLY + PROXY)
        for name, cols in VARIANTS.items():
            unknown = set(cols) - all_known
            self.assertFalse(unknown, f'变体 {name!r} 包含未知列：{unknown}')

    def test_tied_scores_select_exact_budget(self):
        y=np.array([0,1,0,0,1,0,0,0,0,0]);p=np.full(10,.2)
        metrics=evaluate(y,p,.5)
        self.assertEqual(metrics['top20_count'],2)
        self.assertAlmostEqual(metrics['average_precision'],.2)

    def test_duplicate_does_not_use_outcome(self):
        frame=pd.DataFrame({'number':[1,2,3],'title':['one','two','ONE!'],'text':['abc','abc','different'],'y':[0,1,1]})
        kept,removed=deduplicate(frame)
        self.assertEqual(kept.number.tolist(),[1])
        self.assertEqual(len(removed),2)

    def test_threshold_validation(self):
        y=np.array([0,0,1,1]);p=np.array([.01,.03,.05,.09])
        self.assertEqual(select_threshold(y,p),.05)

    def test_landmark_labels_and_comment_time_boundary(self):
        start=pd.Timestamp('2026-05-01T00:00:00Z')
        def issue(n,reason=None,hours=None):
            return {'number':n,'title':f'distinct issue {n}','body':'description',
                    'created_at':str(start),'updated_at':str(start),
                    'closed_at':str(start+pd.Timedelta(hours=hours)) if hours is not None else None,
                    'state':'closed' if hours is not None else 'open','state_reason':reason,
                    'comments':0,'user':'author','author_association':'NONE','labels':[]}
        issues=[issue(1,'completed',240),issue(2,'completed',1),issue(3,'duplicate',240),issue(4),issue(5,'reopened')]
        comments=[]
        for n,h,user in [(1,6,'other'),(2,7,'author'),(3,25,'late')]:
            comments.append({'id':n,'issue_number':1,'created_at':str(start+pd.Timedelta(hours=h)),
                'updated_at':str(start+pd.Timedelta(hours=h)),'body':'reply','user':user,
                'author_association':'NONE','performed_via_github_app':False,'reactions':0})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'issues.json').write_text(json.dumps(issues),encoding='utf-8')
            (root/'comments.json').write_text(json.dumps(comments),encoding='utf-8')
            config={'issues':str(root/'issues.json'),'comments':str(root/'comments.json'),
                'cutoff':'2026-09-13T00:00:00Z','hours':[6,24],'days':[7,14,30],
                'train_end':'2026-04-01T00:00:00Z','validation_start':'2026-05-01T00:00:00Z',
                'validation_end':'2026-06-01T00:00:00Z','test_start':'2026-07-01T00:00:00Z','test_end':'2026-08-01T00:00:00Z'}
            with contextlib.redirect_stdout(io.StringIO()): frames,_,_=prepare(config,root)
            seven=frames[(6,7)].set_index('number'); fourteen=frames[(6,14)].set_index('number')
            self.assertNotIn(2,seven.index);self.assertNotIn(5,seven.index)
            self.assertEqual(seven.loc[1,'y'],0);self.assertEqual(fourteen.loc[1,'y'],1)
            self.assertEqual(fourteen.loc[3,'y'],0);self.assertEqual(fourteen.loc[4,'y'],0)
            self.assertEqual(seven.loc[1,'comment_count'],1)
            self.assertEqual(seven.loc[1,'author_after_other'],0)
            later=frames[(24,14)].set_index('number')
            self.assertEqual(later.loc[1,'comment_count'],2)
            self.assertEqual(later.loc[1,'author_after_other'],1)

if __name__=='__main__':unittest.main()
