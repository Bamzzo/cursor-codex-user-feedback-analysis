"""无需安装即可在项目目录运行 CLI。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from issue_ml.classification.cli import main

if __name__=='__main__':
    main()
