# -*- coding: utf-8 -*-
"""全局路径、口径常量与绘图风格。所有脚本只从这里读取路径。"""
from pathlib import Path
import os

# 项目根目录 = src/common/config.py 的上两级
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.getenv("FEEDBACK_RAW_DIR", ROOT / "data" / "raw"))
DELIVERY_DIR = Path(os.getenv("FEEDBACK_DELIVERY_DIR", ROOT / "data" / "delivery"))
LEGACY_RQ1 = ROOT / "data" / "legacy" / "rq1"
LEGACY_RQ2 = ROOT / "data" / "legacy" / "rq2"
LEGACY_RQ3 = ROOT / "data" / "legacy" / "rq3"

DATA = ROOT / "data"
DERIVED = DATA / "derived"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
PROJECT_DOCS = ROOT / "docs"
REPORTS = PROJECT_DOCS
DOCS = PROJECT_DOCS

for d in [DERIVED, OUT, FIG, PROJECT_DOCS, REPORTS, DOCS,
          OUT / "rq1", OUT / "rq2", OUT / "rq3", OUT / "rq4", OUT / "integrated"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- 口径常量 ----------
SEED = 42
# 观察终点：Issue 表中最新时间字段的推定值，不是已核实的抓取完成时间。
SNAPSHOT_UTC = "2026-09-13T02:20:37Z"
# 渠道比较统一起点：论坛正文从 2026-03 起，但元数据与标签覆盖 2026 全年；
# 主比较使用 2026-01-01 ~ 快照，正文相关分析单独标注 2026-03 起。
COMPARE_START = "2026-01-01"
CENSUS_START = "2026-07-01"
BODY_START = "2026-03-01"

RAW_FILES = {
    "issues": "codex_issues_full.json",
    "comments": "codex_comments.json",
    "forum_topics": "cursor_forum_topics_full.json",
    "forum_content": "cursor_forum_content.json",
    "forum_census": "cursor_forum_content_census.json",
    "hn": "hn_discussions.json",
}

# 论坛版块（与 RQ1 交付一致，来自 site.json 计数校验）
FORUM_CATEGORIES = {
    6: "Bug Reports", 8: "Help", 5: "Feature Requests", 4: "Discussions", 7: "Feedback",
    19: "Built for Cursor", 20: "Guides", 25: "Asia", 26: "Europe", 27: "North America",
    28: "South America", 29: "Africa", 23: "Extension Verification", 10: "Release Discussions",
    16: "Meta", 11: "Announcements", 30: "Oceania", 18: "Built with Cursor", 9: "Showcase",
}

FORUM_GROUP = {
    "Bug Reports": "缺陷报告", "Help": "使用求助", "Feature Requests": "功能需求",
    "Feedback": "体验反馈", "Discussions": "观点讨论",
    "Built for Cursor": "生态与教程", "Guides": "生态与教程", "Built with Cursor": "生态与教程",
    "Showcase": "生态与教程", "Extension Verification": "生态与教程",
    "Release Discussions": "官方公告/版本", "Announcements": "官方公告/版本", "Meta": "官方公告/版本",
    "Asia": "地区社群", "Europe": "地区社群", "North America": "地区社群",
    "South America": "地区社群", "Africa": "地区社群", "Oceania": "地区社群",
}

MAIN4 = ["Bug Reports", "Help", "Feature Requests", "Feedback"]

# 颜色（统一视觉）
COLORS = ["#2563a6", "#dc8551", "#299b89", "#9a6bb0", "#c0504d", "#7f7f7f", "#e2b93b", "#5b9bd5"]

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9,
        "figure.facecolor": "white",
    })
    return plt
