import joblib
from .preprocessing import clean, PREPROCESSING_VERSION


def predict(model_path, title, body):
    """对单条 Issue 进行推理，返回标签和置信度。

    参数
    ----
    model_path : str | Path  joblib 模型包路径。
    title      : str         Issue 标题。
    body       : str         Issue 正文（允许为空字符串）。

    返回
    ----
    dict，包含 label / positive_label / positive_score。
    """
    bundle = joblib.load(model_path)
    if bundle['preprocessing'] != PREPROCESSING_VERSION:
        raise ValueError(
            f'模型清洗版本 {bundle["preprocessing"]!r} 与当前版本 '
            f'{PREPROCESSING_VERSION!r} 不兼容，请重新训练。'
        )
    text = clean(title, body)
    if not text:
        raise ValueError('标题和正文清洗后均为空，无法预测')
    p = float(bundle['pipeline'].predict_proba([text])[0, 1])
    index = int(p >= bundle['threshold'])
    return {
        'label':          bundle['labels'][index],
        'positive_label': bundle['labels'][1],   # labels[1] 约定为正类
        'positive_score': p,
    }
