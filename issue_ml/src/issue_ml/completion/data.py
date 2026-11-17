import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ..classification.preprocessing import clean

TIME      = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'weekend']
STRUCTURE = ['title_length', 'body_length', 'word_count', 'has_code', 'has_log',
             'has_version', 'url_count', 'body_missing']
# EARLY：早期互动计数，不含快照文本、关联身份或机器人/App 分类。
EARLY     = ['comment_count', 'participant_count', 'author_comment_count', 'other_comment_count',
             'other_participant_count', 'has_other', 'first_other_hours', 'author_after_other']
PROXY     = ['external_nonbot_count', 'external_nonbot_participants', 'has_external_nonbot',
             'first_external_nonbot_hours', 'author_after_external_nonbot',
             'bot_proxy_count', 'maintainer_proxy_count']

# 特征变体定义：列出每个变体使用的**结构化列**（不含文本 pipeline）。
# 'combined' 在列上与 'structure_early' 相同；区别在于 experiment.py 中
# CompletionFeatures.fit() 会为 'combined' 额外追加 TF-IDF→SVD 文本表示。
# 此设计将"哪些列"和"是否加文本"分别控制，两者共同决定最终特征矩阵。
VARIANTS = {
    'early_only':      TIME + EARLY,                        # 无快照文本、无结构、无身份代理
    'text_structure':  TIME + STRUCTURE,                    # 文本 SVD + 结构，无早期计数
    'structure_early': TIME + STRUCTURE + EARLY + PROXY,   # 结构 + 早期计数，无文本
    'combined':        TIME + STRUCTURE + EARLY + PROXY,   # 结构 + 早期计数 + 文本 SVD
}

# 需要追加文本 pipeline 的变体集合（由 experiment.py 的 CompletionFeatures 读取）。
TEXT_VARIANTS = frozenset({'text_structure', 'combined'})

def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def _tokens(text):
    return re.findall(r'\b\w+\b', text.lower())

def _simhash(shingles):
    raw = b''.join(hashlib.blake2b(word.encode(), digest_size=8).digest() for word in shingles)
    bits = np.unpackbits(np.frombuffer(raw,dtype=np.uint8).reshape(-1,8),axis=1,bitorder='little')
    mask = bits.sum(axis=0) * 2 >= len(shingles)
    return int.from_bytes(np.packbits(mask,bitorder='little').tobytes(),'little')

def deduplicate(frame):
    """按创建时间保留最早记录，不使用目标标签决定去重。"""
    exact, titles, bands, accepted, removed = set(), set(), defaultdict(list), {}, []
    keep = []
    for idx, row in frame.iterrows():
        text = re.sub(r'\s+', ' ', row.text).strip().lower()
        title = re.sub(r'\W+', ' ', row.title.lower()).strip()
        reason, match = '', None
        if text in exact:
            reason = 'exact_cleaned_text'
        elif title and title in titles:
            reason = 'same_normalized_title'
        words = _tokens(text)
        shingles = {' '.join(words[i:i+3]) for i in range(len(words)-2)} if len(words) >= 30 else set()
        sim = _simhash(shingles) if shingles and not reason else None
        if sim is not None:
            candidates = set(j for b in range(4) for j in bands[(b, (sim >> (b*16)) & 65535)])
            for j in candidates:
                other_sim, other_shingles, number = accepted[j]
                if (sim ^ other_sim).bit_count() <= 3 and len(shingles & other_shingles) / len(shingles | other_shingles) >= .85:
                    reason, match = 'lexical_near_duplicate', number
                    break
        if reason:
            removed.append({'number':int(row.number), 'reason':reason, 'matched_number':match})
            continue
        keep.append(idx); exact.add(text)
        if title: titles.add(title)
        if sim is not None:
            accepted[idx] = (sim, shingles, int(row.number))
            for b in range(4): bands[(b, (sim >> (b*16)) & 65535)].append(idx)
    return frame.loc[keep].copy(), pd.DataFrame(removed, columns=['number','reason','matched_number'])

def prepare(config, output):
    output = Path(output)
    raw, issue_hash = read_json(config['issues'])
    comments, comment_hash = read_json(config['comments'])
    a, c = pd.DataFrame(raw), pd.DataFrame(comments)
    cutoff = pd.Timestamp(config['cutoff'])
    audit = {'issues_raw':len(a), 'comments_raw':len(c), 'issue_sha256':issue_hash, 'comment_sha256':comment_hash,
             'analysis_cutoff':config['cutoff'], 'cutoff_status':'conservative analytic cutoff, not verified collection timestamp'}
    for col in ['created_at','updated_at','closed_at']:
        a[col] = pd.to_datetime(a[col], utc=True, errors='coerce')
    audit['issue_time_maxima'] = {k:str(a[k].max()) for k in ['created_at','updated_at','closed_at']}
    audit['issue_duplicate_ids'] = int(a.number.duplicated().sum())
    a = a.sort_values('updated_at').drop_duplicates('number', keep='last')
    valid = a.created_at.notna() & ((a.closed_at.isna()) | (a.closed_at >= a.created_at))
    valid &= (a.state != 'closed') | a.closed_at.notna()
    audit['invalid_time_or_closed_without_date'] = int((~valid).sum())
    audit['created_after_cutoff'] = int((a.created_at > cutoff).sum())
    audit['known_reopened'] = int(a.state_reason.eq('reopened').sum())
    a = a[valid & a.created_at.le(cutoff) & ~a.state_reason.eq('reopened')].copy()
    audit['after_time_and_reopened_filter'] = len(a)
    a['title'] = a.title.fillna(''); a['body'] = a.body.fillna('')
    a['text'] = [clean(t,b) for t,b in zip(a.title,a.body)]
    audit['empty_cleaned_text'] = int(a.text.str.strip().eq('').sum())
    a = a[a.text.str.strip().ne('')].sort_values(['created_at','number'])
    print('Deduplicating exact/title/lexical-near records...', flush=True)
    a, duplicate_rows = deduplicate(a)
    duplicate_rows.to_csv(output/'duplicate_exclusions.csv',index=False,encoding='utf-8-sig')
    audit['duplicate_exclusions'] = duplicate_rows.reason.value_counts().to_dict()
    audit['after_deduplication'] = len(a)
    a['age_days'] = (cutoff-a.created_at).dt.total_seconds()/86400
    a['close_hours'] = (a.closed_at-a.created_at).dt.total_seconds()/3600
    a['observed_closed'] = a.state.eq('closed') & a.closed_at.le(cutoff)
    a['title_length'] = a.title.str.len(); a['body_length'] = a.body.str.len()
    a['word_count'] = a.body.str.split().str.len()
    a['has_code'] = a.body.str.contains(r'```|<code',regex=True).astype(int)
    a['has_log'] = a.body.str.contains(r'(?i)traceback|stack trace|exception|error\s*[:\[]',regex=True).astype(int)
    a['has_version'] = a.body.str.contains(r'\b\d+\.\d+(?:\.\d+)?\b',regex=True).astype(int)
    a['url_count'] = a.body.str.count(r'https?://')
    a['body_missing'] = a.body.str.strip().eq('').astype(int)
    hour, weekday = a.created_at.dt.hour, a.created_at.dt.weekday
    for name, values, period in [('hour',hour,24),('weekday',weekday,7)]:
        a[name+'_sin'] = np.sin(2*np.pi*values/period); a[name+'_cos'] = np.cos(2*np.pi*values/period)
    a['weekend'] = weekday.ge(5).astype(int)
    # 关联完整性使用原始父表，避免把去重排除误计为评论缺失。
    all_numbers = {x['number'] for x in raw}
    audit['comment_duplicate_ids'] = int(c.id.duplicated().sum())
    c['created_at'] = pd.to_datetime(c.created_at,utc=True,errors='coerce')
    c['updated_at'] = pd.to_datetime(c.updated_at,utc=True,errors='coerce')
    c = c.sort_values('updated_at').drop_duplicates('id',keep='last')
    audit['comment_matched_raw'] = int(c.issue_number.isin(all_numbers).sum())
    audit['comment_unmatched_raw'] = int((~c.issue_number.isin(all_numbers)).sum())
    counts = c.groupby('issue_number').size()
    diffs = pd.Series({x['number']:int(counts.get(x['number'],0))-x['comments'] for x in raw})
    audit['parent_comment_count_equal'] = int(diffs.eq(0).sum())
    audit['parent_comment_count_deficit'] = int(diffs.lt(0).sum())
    audit['parent_comment_count_excess'] = int(diffs.gt(0).sum())
    a['comment_count_matches'] = a.number.map(diffs).eq(0)
    parent = pd.DataFrame(raw)[['number','user','created_at']].rename(columns={'user':'issue_user','created_at':'issue_created'})
    parent['issue_created'] = pd.to_datetime(parent.issue_created,utc=True)
    c = c.merge(parent,left_on='issue_number',right_on='number',how='inner',validate='many_to_one')
    c['delay_hours'] = (c.created_at-c.issue_created).dt.total_seconds()/3600
    audit['comments_before_issue'] = int(c.delay_hours.lt(0).sum())
    audit['comments_after_cutoff'] = int(c.created_at.gt(cutoff).sum())
    audit['comments_invalid_time'] = int(c.created_at.isna().sum())
    c = c[c.delay_hours.ge(0) & c.created_at.le(cutoff)].copy()
    c['self'] = c.user.eq(c.issue_user)
    c['bot'] = c.user.str.contains(r'(?i)\[bot\]$|-bot$',regex=True,na=False) | c.performed_via_github_app.fillna(False).astype(bool)
    c['external'] = ~c['self'] & ~c.bot
    c['maintainer'] = c.external & c.author_association.isin(['MEMBER','COLLABORATOR','OWNER'])
    c = c[c.issue_number.isin(a.number) & c.delay_hours.le(max(config['hours']))]
    groups = {int(n):g for n,g in c.groupby('issue_number')}
    frames, flow = {}, []
    for h in config['hours']:
        records=[]
        for row in a.itertuples():
            g = groups.get(row.number)
            g = g[g.delay_hours.le(h)] if g is not None else c.iloc[:0]
            other = g[~g['self']]; ext = g[g.external]
            first_other = float(other.delay_hours.min()) if len(other) else float(h)
            first_ext = float(ext.delay_hours.min()) if len(ext) else float(h)
            records.append({'number':row.number,'comment_count':len(g),'participant_count':g.user.nunique(),
                'author_comment_count':int(g['self'].sum()),'other_comment_count':len(other),'other_participant_count':other.user.nunique(),
                'has_other':int(len(other)>0),'first_other_hours':first_other,
                'author_after_other':int(len(other)>0 and (g['self'] & g.delay_hours.gt(first_other)).any()),
                'external_nonbot_count':len(ext),'external_nonbot_participants':ext.user.nunique(),'has_external_nonbot':int(len(ext)>0),
                'first_external_nonbot_hours':first_ext,'author_after_external_nonbot':int(len(ext)>0 and (g['self'] & g.delay_hours.gt(first_ext)).any()),
                'bot_proxy_count':int(g.bot.sum()),'maintainer_proxy_count':int(g.maintainer.sum())})
        feature_frame = a.merge(pd.DataFrame(records),on='number',validate='one_to_one')
        for d in config['days']:
            mature = feature_frame[feature_frame.age_days.ge(d)]
            cohort = mature[~(mature.observed_closed & mature.close_hours.le(h))].copy()
            cohort['y'] = (cohort.observed_closed & cohort.state_reason.eq('completed') & cohort.close_hours.gt(h) & cohort.close_hours.le(d*24)).astype(int)
            cohort['split'] = assign_split(cohort,config,d)
            frames[(h,d)] = cohort
            flow.append({'h':h,'d':d,'after_deduplication':len(a),'mature_d':len(mature),
                'closed_by_h_removed':len(mature)-len(cohort),'eligible':len(cohort),'positives':int(cohort.y.sum()),'rate':float(cohort.y.mean()),
                **{f'{s}_n':int(cohort.split.eq(s).sum()) for s in ['train','validation','test','unused']}})
    flow = pd.DataFrame(flow); flow.to_csv(output/'sample_flow.csv',index=False,encoding='utf-8-sig')
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(flow.to_string(index=False),flush=True)
    return frames, audit, flow

def assign_split(frame,config,d):
    train_end, val_start, val_end, test_start, test_end = [pd.Timestamp(config[k]) for k in ['train_end','validation_start','validation_end','test_start','test_end']]
    if train_end + pd.Timedelta(days=d) > val_start or val_end + pd.Timedelta(days=d) > test_start:
        raise ValueError('Training or validation labels have not matured before the next phase')
    result = pd.Series('unused',index=frame.index)
    result.loc[frame.created_at.lt(train_end)] = 'train'
    result.loc[frame.created_at.ge(val_start) & frame.created_at.lt(val_end)] = 'validation'
    result.loc[frame.created_at.ge(test_start) & frame.created_at.lt(test_end)] = 'test'
    return result
