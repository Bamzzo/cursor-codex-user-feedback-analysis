"""Issue completed 预测独立入口。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from issue_ml.completion.__main__ import main

if __name__=='__main__':main()
