import json
from pathlib import Path

_REQUIRED_TASK_KEYS   = ('negative_label', 'positive_label')
_REQUIRED_SPLIT_KEYS  = ('train_fraction',)
_REQUIRED_CONFIG_KEYS = ('source', 'output', 'seed', 'threshold', 'task', 'split', 'features', 'model')


def load_config(path):
    """读取 Task A（bug/enhancement 分类）的 JSON 配置，解析并验证所有必填字段。

    - 路径相对于配置文件所在目录下的 project_root 解析，不依赖运行时工作目录。
    - 返回 resolved config（所有路径均为绝对路径字符串）。
    """
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    root = (path.parent / config.get('project_root', '..')).resolve()

    # 检查顶层必填字段
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f'配置缺少必填字段：{missing}')

    # 解析路径
    for field in ('source', 'output'):
        config[field] = str((root / config[field]).resolve())

    # 验证 task
    task = config['task']
    missing_task = [k for k in _REQUIRED_TASK_KEYS if k not in task]
    if missing_task:
        raise KeyError(f'task 缺少必填字段：{missing_task}')
    if task['negative_label'] == task['positive_label']:
        raise ValueError('task.negative_label 和 task.positive_label 不能相同')

    # 验证 split
    missing_split = [k for k in _REQUIRED_SPLIT_KEYS if k not in config['split']]
    if missing_split:
        raise KeyError(f'split 缺少必填字段：{missing_split}')
    if not 0 < config['split']['train_fraction'] < 1:
        raise ValueError('split.train_fraction 必须在 (0, 1) 范围内')

    # 验证阈值
    if not 0 < config['threshold'] < 1:
        raise ValueError('threshold 必须在 (0, 1) 范围内')

    # 验证模型名称（早期检查，避免数据加载后才报错）
    model_name = config.get('model', {}).get('name', '')
    if model_name != 'logistic_regression':
        raise ValueError(
            f'不支持的模型类型 {model_name!r}，当前仅支持 logistic_regression'
        )

    return config
