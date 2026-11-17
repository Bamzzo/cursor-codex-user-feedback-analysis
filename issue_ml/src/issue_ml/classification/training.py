import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data import prepare
from .splitting import temporal_split
from .modeling import build_pipeline, TFIDF_STEP, LR_STEP
from .evaluation import metrics
from .reporting import write_report
from .preprocessing import PREPROCESSING_VERSION


def run(config):
    """Task A 训练主流程：加载数据 → 划分 → 训练 → 评估 → 保存产物。

    输出目录非空时拒绝运行，避免覆盖已有实验。每次运行结果独立保存在 config['output']。
    """
    output = Path(config['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'输出目录已有内容：{output}\n请修改配置中的 output 字段，使用新目录。'
        )

    # --- 数据加载与指纹 ---
    raw_bytes = Path(config['source']).read_bytes()
    labels = [config['task']['negative_label'], config['task']['positive_label']]
    # labels[0] = 负类，labels[1] = 正类；prediction.py 依赖此顺序。

    frame, audit = prepare(json.loads(raw_bytes), *labels)
    train, test, split_info = temporal_split(frame, config['split']['train_fraction'])

    if train.y.nunique() != 2 or test.y.nunique() != 2:
        raise ValueError('训练集和测试集中均需包含两个类别')

    print(f'Training {len(train):,}; testing {len(test):,}', flush=True)

    # --- 训练与评估 ---
    model = build_pipeline(config)
    model.fit(train.text, train.y)
    probabilities = model.predict_proba(test.text)[:, 1]
    threshold = config['threshold']
    prior = np.full(len(test), float(train.y.mean()))

    result = {
        **audit,
        **split_info,                                             # time_boundary 已是 str
        'train_n':             len(train),
        'test_n':              len(test),
        'source':              config['source'],
        'sha256':              hashlib.sha256(raw_bytes).hexdigest(),
        'train_dates':         [str(train.created_at.min()), str(train.created_at.max())],
        'test_dates':          [str(test.created_at.min()),  str(test.created_at.max())],
        'train_positive_rate': float(train.y.mean()),
        'test_positive_rate':  float(test.y.mean()),
        'baseline':            metrics(test.y, prior, threshold, labels),
        'logistic_regression': metrics(test.y, probabilities, threshold, labels),
        'versions': {
            'python':  platform.python_version(),
            'sklearn': sklearn.__version__,
        },
    }

    # --- 保存产物 ---
    output.mkdir(parents=True, exist_ok=True)

    # 数据集
    for name, data in [('cleaned_dataset', frame), ('train', train), ('test', test)]:
        data[['number', 'created_at', 'title', 'text', 'y']].to_json(
            output / f'{name}.jsonl', orient='records', lines=True, force_ascii=False
        )

    # 模型包（labels[1] 为正类约定写入 bundle）
    bundle = {
        'pipeline':       model,
        'labels':         labels,
        'threshold':      threshold,
        'preprocessing':  PREPROCESSING_VERSION,
        'config':         config,
    }
    joblib.dump(bundle, output / 'model.joblib')

    # 指标与配置
    for name, data in [('metrics', result), ('config', config)]:
        (output / f'{name}.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    # 预测与误判
    score_col = f'p_{labels[1]}'
    preds = test[['number', 'created_at', 'title', 'y', 'text']].copy()
    preds['prediction'] = (probabilities >= threshold).astype(int)
    preds[score_col] = probabilities
    preds.to_csv(output / 'test_predictions.csv', index=False, encoding='utf-8-sig')

    errors = preds[preds.y != preds.prediction].copy()
    errors['confidence'] = np.maximum(errors[score_col], 1 - errors[score_col])
    errors.sort_values('confidence', ascending=False).to_csv(
        output / 'errors.csv', index=False, encoding='utf-8-sig'
    )

    # 样本划分记录
    pd.concat([
        train[['number', 'y', 'created_at']].assign(split='train'),
        test[['number',  'y', 'created_at']].assign(split='test'),
    ]).to_csv(output / 'sample_split.csv', index=False, encoding='utf-8-sig')

    # Top 特征词（通过常量引用步骤名，避免魔法字符串）
    terms = pd.DataFrame({
        'term':        model[TFIDF_STEP].get_feature_names_out(),
        'coefficient': model[LR_STEP].coef_[0],
    })
    pd.concat([
        terms.nsmallest(20, 'coefficient'),
        terms.nlargest(20,  'coefficient'),
    ]).to_csv(output / 'top_terms.csv', index=False, encoding='utf-8-sig')

    write_report(result, output, labels)
    print(json.dumps(result['logistic_regression'], ensure_ascii=False, indent=2))
    return result
