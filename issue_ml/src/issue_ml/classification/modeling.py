from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pipeline 步骤名常量，由 splitting.py 导出并在 training.py 中使用，
# 避免两处硬编码字符串 'tfidf' / 'lr' 相互耦合。
TFIDF_STEP = 'tfidf'
LR_STEP    = 'lr'

_SUPPORTED_MODELS = {'logistic_regression'}


def build_pipeline(config):
    """根据配置构建 sklearn Pipeline（TF-IDF + 分类器）。

    仅支持 logistic_regression。若需新增模型，在此函数中增加显式分支并保留
    统一的 fit / predict_proba 接口。
    """
    features = dict(config['features'])
    features['ngram_range'] = tuple(features['ngram_range'])

    model_cfg = dict(config['model'])
    name = model_cfg.pop('name')
    if name not in _SUPPORTED_MODELS:
        raise ValueError(
            f'不支持的模型类型 {name!r}，当前支持：{sorted(_SUPPORTED_MODELS)}'
        )

    return Pipeline([
        (TFIDF_STEP, TfidfVectorizer(**features)),
        (LR_STEP,    LogisticRegression(random_state=config['seed'], **model_cfg)),
    ])
