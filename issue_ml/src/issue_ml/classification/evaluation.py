from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score,
)
import numpy as np


def metrics(y, p, threshold=0.5, labels=("bug", "enhancement")):
    """计算二分类评估指标。

    参数
    ----
    y         : array-like  真实标签（0/1）。
    p         : array-like  正类预测概率。
    threshold : float       将概率转为标签的阈值。
    labels    : tuple[str]  (负类名, 正类名)，用于 per_class 报告的 target_names。

    返回
    ----
    dict，包含 accuracy / macro_f1 / roc_auc / average_precision /
              confusion_matrix / per_class。
    当类别数不足 2 时，roc_auc 和 average_precision 返回 None。
    """
    y    = np.asarray(y, dtype=int)
    p    = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    n_classes = len(np.unique(y))
    roc_auc           = float(roc_auc_score(y, p))           if n_classes == 2 else None
    average_precision = float(average_precision_score(y, p)) if n_classes == 2 else None

    return {
        'accuracy':           float(accuracy_score(y, pred)),
        'macro_f1':           float(f1_score(y, pred, average='macro')),
        'roc_auc':            roc_auc,
        'average_precision':  average_precision,
        'confusion_matrix':   confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'per_class':          classification_report(
                                  y, pred, labels=[0, 1],
                                  target_names=list(labels),
                                  output_dict=True, zero_division=0,
                              ),
    }
