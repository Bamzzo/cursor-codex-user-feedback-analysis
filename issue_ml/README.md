## 项目结构

```text
issue_ml/
├── configs/                   # 本地配置（按需创建，不含原始数据）
├── data/raw/                  # 原始 JSON，只读输入，不提交 Git
├── src/issue_ml/
│   ├── classification/        # 任务 A：bug / enhancement 分类
│   │   ├── config.py          # 配置读取与路径解析
│   │   ├── preprocessing.py   # 训练与预测共用文本清洗
│   │   ├── data.py            # 标签筛选、冲突处理、文本去重
│   │   ├── splitting.py       # 按时间划分、排除跨集合重复标题
│   │   ├── modeling.py        # 特征与分类器 Pipeline 工厂
│   │   ├── training.py        # 训练流程、数据指纹与产物保存
│   │   ├── evaluation.py      # 分类指标与基线
│   │   ├── reporting.py       # 混淆矩阵与结果报告
│   │   ├── prediction.py      # 单条 Issue 推理
│   │   └── cli.py             # train / predict 命令
│   └── completion/            # 任务 B：Issue completed 预测（随机森林）
├── tests/                     # 清洗、标签范围与时间划分契约检查
├── artifacts/                 # 本地实验产物，不提交 Git
├── docs/architecture.md       # 数据流和扩展约定
├── run.py                     # 无需安装的本地入口
└── pyproject.toml             # Python 包及依赖声明
```

公开仓库保留了扩展代码、测试和已整理的结果说明，但不包含原始 Issue 数据、完整训练产物或内部配置文件。使用前请在本地准备输入数据，并按代码中的路径配置运行。

