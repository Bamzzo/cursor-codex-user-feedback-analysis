from .preprocessing import normalise

# Pipeline 步骤名常量，training.py 通过这些名称访问子步骤，避免魔法字符串散落各处。
TFIDF_STEP = 'tfidf'
LR_STEP    = 'lr'


def temporal_split(df, train_fraction=0.8):
    """按创建时间对 df 做时序划分，返回 (train, test, split_info)。

    - 以第 int(n*train_fraction) 条记录的 created_at 为边界；边界值及之后属于测试集。
    - 训练集中出现过的规范化标题在测试集中会被移除，以防止跨集信息泄露。
    - 所有完整性检查均抛出 ValueError 而非依赖 assert，以便在 -O 模式下同样生效。
    """
    if not 0 < train_fraction < 1 or len(df) < 2:
        raise ValueError('Need at least two records and 0 < train_fraction < 1')

    boundary = df.created_at.iloc[int(len(df) * train_fraction)]
    train_pool = df[df.created_at < boundary].copy()
    test_pool  = df[df.created_at >= boundary].copy()

    # 防止标题相同而正文略有变化的记录跨越训练/测试边界。
    train_titles = set(train_pool.title.map(normalise))
    title_overlap = int(test_pool.title.map(normalise).isin(train_titles).sum())
    test_pool = test_pool[~test_pool.title.map(normalise).isin(train_titles)]

    train = train_pool.sort_values('created_at').copy()
    test  = test_pool.sort_values('created_at').copy()

    if train.empty or test.empty:
        raise ValueError('Empty train/test set after temporal split')

    # 完整性检查：时间隔离、编号不重叠、text_key 不重叠、标题不重叠、文本非空。
    if not (train.created_at.max() < test.created_at.min()):
        raise ValueError('时间划分完整性检查失败：训练集最大时间 >= 测试集最小时间')
    if not set(train.number).isdisjoint(test.number):
        raise ValueError('时间划分完整性检查失败：编号存在重叠')
    if not set(train.text_key).isdisjoint(test.text_key):
        raise ValueError('时间划分完整性检查失败：text_key 存在重叠')
    if not set(train.title.map(normalise)).isdisjoint(test.title.map(normalise)):
        raise ValueError('时间划分完整性检查失败：规范化标题存在重叠')
    if not (train.text.str.strip().ne('').all() and test.text.str.strip().ne('').all()):
        raise ValueError('时间划分完整性检查失败：存在清洗后为空的文本')

    return train, test, {
        'time_boundary':             str(boundary),   # 序列化为字符串，兼容 JSON
        'test_title_overlap_removed': title_overlap,
    }
