import argparse
import json
import sys
from .config import load_config


def main():
    """Task A CLI 入口：train 子命令训练模型，predict 子命令对单条 Issue 推理。"""
    parser = argparse.ArgumentParser(description='Issue 文本分类实验（bug / enhancement）')
    sub = parser.add_subparsers(dest='command', required=True)

    train_p = sub.add_parser('train', help='清洗、划分、训练并输出评估结果')
    train_p.add_argument('--config', required=True, help='配置文件路径（JSON）')

    pred_p = sub.add_parser('predict', help='对一条新 Issue 进行分类')
    pred_p.add_argument('--model',  required=True, help='模型包路径（.joblib）')
    pred_p.add_argument('--title',  required=True, help='Issue 标题')
    pred_p.add_argument('--body',   default='',    help='Issue 正文（默认为空）')

    args = parser.parse_args()

    if args.command == 'train':
        try:
            from .training import run
            run(load_config(args.config))
        except (FileNotFoundError, KeyError, ValueError, FileExistsError) as exc:
            print(f'[错误] {exc}', file=sys.stderr)
            sys.exit(1)
    else:
        try:
            from .prediction import predict
            print(json.dumps(predict(args.model, args.title, args.body),
                             ensure_ascii=False, indent=2))
        except (FileNotFoundError, ValueError) as exc:
            print(f'[错误] {exc}', file=sys.stderr)
            sys.exit(1)
