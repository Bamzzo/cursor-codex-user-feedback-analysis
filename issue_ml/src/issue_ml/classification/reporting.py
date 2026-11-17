from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _fmt(value):
    """将指标值格式化为字符串；None 显示为 'N/A'。"""
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def write_report(result, output, labels):
    """将混淆矩阵图和 Markdown 实验报告写入 output 目录。

    参数
    ----
    result : dict   training.run() 返回的完整结果字典。
    output : Path   产物输出目录。
    labels : list   [负类名, 正类名]，用于图表坐标轴。
    """
    output = Path(output)
    m = result['logistic_regression']
    b = result['baseline']
    cm = np.array(m['confusion_matrix'])

    # --- 混淆矩阵图 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        xlabel='预测标签', ylabel='参考标签',
        title=f'测试集  n={result["test_n"]:,}',
    )
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)

    # --- Markdown 报告 ---
    lines = [
        '# 实验结果',
        '',
        f'训练 {result["train_n"]:,} 条；测试 {result["test_n"]:,} 条。',
        '',
        '|指标|先验基线|模型|',
        '|---|---:|---:|',
    ]
    for metric_key in ('accuracy', 'macro_f1', 'roc_auc', 'average_precision'):
        lines.append(f'|{metric_key}|{_fmt(b[metric_key])}|{_fmt(m[metric_key])}|')

    lines += [
        '',
        f'判对 {int(cm.trace())} 条，判错 {int(cm.sum() - cm.trace())} 条。',
        '',
        '该结果衡量与平台快照标签的一致性。快照正文可能经过编辑；语义近重复与重复作者未完全排除。',
        '清洗会删除所有 Markdown 标题行，包括用户自拟标题；保留此规则是为了与原实验一致。',
        '无标签和同时含两类标签的 Issue 不属于本次评估范围。',
    ]
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
