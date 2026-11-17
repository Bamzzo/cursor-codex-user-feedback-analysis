import pandas as pd
from .preprocessing import clean, normalise

# 要求输入记录中必须存在的字段。
_REQUIRED_FIELDS = ('number', 'created_at', 'title', 'body', 'labels')

def prepare(raw, negative, positive):
    """从原始 Issue 列表中筛选出恰好含一个目标标签且正文非空的记录。

    参数
    ----
    raw      : list[dict]  原始 Issue 记录，每条必须包含 _REQUIRED_FIELDS 中的字段。
    negative : str         负类标签名（如 'bug'）。
    positive : str         正类标签名（如 'enhancement'）。

    返回
    ----
    df    : DataFrame  清洗、去重后的记录，含 text_key 列。
    audit : dict       数据处理统计。
    """
    if negative == positive:
        raise ValueError(f'negative_label 和 positive_label 不能相同：{negative!r}')

    rows = []
    for i, x in enumerate(raw):
        for field in _REQUIRED_FIELDS:
            if field not in x:
                raise KeyError(f'记录索引 {i} 缺少必填字段 {field!r}')
        labels = set(x['labels'])
        # 恰好含一个目标标签，且正文非空
        if (negative in labels) == (positive in labels):
            continue
        if not (x['body'] or '').strip():
            continue
        rows.append({
            'number':     x['number'],
            'created_at': x['created_at'],
            'title':      x['title'],
            'text':       clean(x['title'], x['body']),
            'y':          int(positive in labels),
        })

    if not rows:
        raise ValueError('筛选后无合格记录，请检查标签名称和数据来源')

    df = pd.DataFrame(rows).sort_values(['created_at', 'number'])
    eligible = len(df)
    df['text_key'] = df.text.map(normalise)

    # 完全相同清洗文本但标签冲突的组全部排除；无冲突时保留最早记录。
    conflicts = df.groupby('text_key').y.nunique()
    conflict_keys = set(conflicts[conflicts > 1].index)
    conflict_n = int(df.text_key.isin(conflict_keys).sum())
    df = df[~df.text_key.isin(conflict_keys)].drop_duplicates('text_key', keep='first')

    return df, {
        'eligible_nonempty':            eligible,
        'conflicting_text_rows_removed': conflict_n,
        'after_text_dedupe':            len(df),
    }
