# -*- coding: utf-8 -*-
"""统一的“入口—能力—环境—问题类型—处理结果”分类字典。

设计原则
--------
1. 每个维度允许多标签；报告分布时同时给出“主类”（primary）与“多标签”（multi）。
2. 无法可靠映射时保留 unknown，不做强制归类。
3. 标签/版块属于维护者或平台的事前分类，文本规则属于研究者的可复核规则；
   `source` 字段记录每个维度来自 label/tag/text/category 中的哪一种。
4. 关联规则和管理型标签不进入“问题类型”结论。
"""
from __future__ import annotations
import re
from typing import Iterable

# --------------------------------------------------------------------------
# 维度定义（顺序即主类优先级，越靠前越具体）
# --------------------------------------------------------------------------
ENTRY_DEF = {
    "app-server": ("应用服务端", "app-server / 本地服务进程 / 端口与中继"),
    "tui": ("TUI", "终端交互界面"),
    "cli": ("CLI", "命令行入口、codex exec"),
    "ios": ("iOS / 移动端", "iPhone、iPad、Android、移动 App"),
    "extension": ("IDE 扩展", "VS Code / JetBrains / 插件"),
    "sdk": ("SDK / API", "SDK、API、GitHub Action"),
    "web": ("Web", "codex-web、网页版界面"),
    "app": ("桌面应用", "文本明确提及 app 或 desktop；品牌词和界面部件不单独推断入口"),
    "unknown": ("未知入口", "公开文本与标签均无法判定"),
}
ENTRY_ORDER = list(ENTRY_DEF)

CAP_DEF = {
    "mcp": ("MCP / 连接器", "MCP server、connector、扩展宿主"),
    "remote": ("远程 / 云端", "remote、SSH、cloud agents、远程控制"),
    "sandbox": ("沙箱与权限", "sandbox、审批、权限、安全策略"),
    "session": ("会话与上下文", "session、thread、history、context、checkpoint"),
    "tool-calls": ("工具调用 / 执行", "tool calls、shell、exec、patch、terminal"),
    "models": ("模型与路由", "模型选择、BYOK、custom model、model behavior"),
    "usage": ("订阅与用量", "billing、quota、rate limit、plan、pricing"),
    "auth": ("登录与账户", "login、auth、OAuth、account"),
    "connectivity": ("网络连通性", "network、proxy、connection、timeout、offline"),
    "agent": ("Agent 工作流", "agent、subagent、composer、plan mode、automation"),
    "code-review": ("代码审查", "review、bugbot、diff review"),
    "ui": ("界面与交互", "layout、window、sidebar、editor、theme、chat UI"),
    "config": ("配置与规则", "config、settings、rules、hooks、keybindings、skills"),
    "workspace": ("工作区与版本控制", "worktree、git、diff、workspace"),
    "bot": ("Bot / 外部集成", "bot、Slack、Linear、集成"),
    "deployment": ("企业 / 云部署", "Azure、Bedrock、enterprise、on-prem"),
    "other": ("其他能力", "无法归入以上能力模块"),
    "unknown": ("未知能力", "无可用信号"),
}
CAP_ORDER = list(CAP_DEF)

ENV_DEF = {
    "windows": ("Windows", "Windows 桌面 / Windows NT"),
    "macos": ("macOS", "macOS / Mac"),
    "linux": ("Linux", "Linux 桌面 / 服务器"),
    "wsl": ("WSL", "Windows Subsystem for Linux"),
    "browser": ("浏览器", "浏览器环境 / Web"),
    "mobile": ("移动系统", "iOS / Android 运行环境"),
    "cloud": ("云端", "云端运行 / 托管环境"),
    "unknown": ("未知环境", "文本与标签均未给出运行环境"),
}
ENV_ORDER = list(ENV_DEF)

TYPE_DEF = {
    "bug": ("缺陷 / 故障", "crash、error、broken、regression、not working"),
    "performance": ("性能问题", "slow、latency、CPU / memory、卡顿"),
    "connectivity": ("连接问题", "connection、network、proxy、timeout"),
    "help": ("使用求助", "how-to、question、need help"),
    "enhancement": ("功能需求", "feature request、please add、suggestion"),
    "feedback": ("体验反馈", "UX、confusing、annoying、improvement"),
    "docs": ("文档问题", "documentation、guide、example"),
    "other": ("其他 / 无法判定", "仅讨论、公告、社区活动等"),
    "unknown": ("未知类型", "无可用信号"),
}
TYPE_ORDER = list(TYPE_DEF)

OUTCOME_DEF = {
    "open": ("仍开启", "snapshot 时 state=open 且 state_reason 不是 reopened"),
    "completed": ("标记完成", "state=closed 且 state_reason=completed"),
    "duplicate": ("重复关闭", "state=closed 且 state_reason=duplicate"),
    "not_planned": ("不计划处理", "state=closed 且 state_reason=not_planned"),
    "reopened": ("已重开", "state_reason=reopened（通常 state=open）；与普通 open 分开计数"),
    "closed_unspecified": ("关闭未注明原因", "state=closed 且 state_reason 缺失"),
    "accepted": ("论坛已采纳", "论坛 has_accepted_answer=True（仅论坛）"),
    "no_reply": ("论坛零回复", "reply_count=0（仅论坛）"),
}
OUTCOME_ORDER = list(OUTCOME_DEF)

# 管理型标签：反映维护流程，不进入产品问题结论
MANAGEMENT_PREFIXES = ("papercuts", "affected:", "source:", "product area:", "owner/")
MANAGEMENT_LABELS = {
    "user request", "feature", "ui polish", "customer blocked", "high impact", "moderate",
    "sev3", "gtms-sev4", "next weekly review", "ready", "normal queue", "needs-response",
    "top 100 customer", "tibo tweet", "selection", "source - slack", "dogfooding",
}

# --------------------------------------------------------------------------
# 标签 → 统一维度（Issue labels / 论坛 tags 共用别名表）
# --------------------------------------------------------------------------
LABEL_MAP: dict[str, dict] = {}


def _add(names, **kw):
    for n in names:
        LABEL_MAP[n.casefold().strip()] = kw


# 产品入口
_add(["app"], entry=["app"])
_add(["cli"], entry=["cli"])
_add(["tui"], entry=["tui"])
_add(["ios", "mobile", "android"], entry=["ios"])
_add(["codex-web"], entry=["web"])
_add(["app-server"], entry=["app-server"])
_add(["extension", "jetbrains"], entry=["extension"])
_add(["sdk", "github-action"], entry=["sdk"])

# 运行环境
_add(["windows-os", "windows"], env=["windows"])
_add(["macos"], env=["macos"])
_add(["linux"], env=["linux"])
_add(["wsl & terminal", "wsl"], env=["wsl"])
_add(["cloud"], env=["cloud"], cap=["remote"])
_add(["browser", "browser use"], cap=["ui"], env=["browser"])

# 能力模块
_add(["mcp", "mcp-server", "connectors"], cap=["mcp"])
_add(["remote"], cap=["remote"])
_add(["sandbox", "safety-check", "tcc / permissions", "sandbox & security"], cap=["sandbox"])
_add(["session", "context", "context mgmt", "memory", "checkpoints"], cap=["session"])
_add(["tool-calls", "exec", "terminal", "language-servers"], cap=["tool-calls"])
_add(["model-behavior", "custom-model", "composer", "codex core"], cap=["models"])
_add(["rate-limits", "enterprise"], cap=["usage"])
_add(["auth", "login"], cap=["auth"])
_add(["connectivity", "networking"], cap=["connectivity"], type=["connectivity"])
_add(["agent", "subagent", "automations", "automation", "plan", "skills", "hooks", "pets", "artifacts-in-codex", "codex-imagegen", "imagen", "aardvark"], cap=["agent"])
_add(["code-review"], cap=["code-review"])
_add(["sidebar", "settings", "chat thread ux", "layout", "themes", "keybindings"], cap=["ui"])
_add(["config", "rules", "plugins", "marketplace"], cap=["config"])
_add(["worktrees", "git", "diffs"], cap=["workspace"])
_add(["slack", "linear"], cap=["bot"])
_add(["azure", "aws-bedrock"], cap=["deployment"])

# 问题类型（仅明确的问题类型标签）
_add(["bug", "regression", "bug / regression", "broken flow", "reliability & trust"], type=["bug"])
_add(["enhancement", "feature request / enhancement", "productivity"], type=["enhancement"])
_add(["documentation", "markdown"], type=["docs"])
_add(["performance", "stability & performance"], cap=["tool-calls"], type=["performance"])
_add(["slow interaction"], cap=["ui"], type=["performance"])
_add(["confusing interaction"], cap=["ui"], type=["feedback"])
_add(["accessibility"], type=["other"])

# 管理型标签（仍保留，但标记 management=True）
for n in MANAGEMENT_LABELS:
    LABEL_MAP.setdefault(n.casefold(), {}).setdefault("management", True)


def label_dimensions(label: str) -> dict:
    key = str(label).casefold().strip()
    d = dict(LABEL_MAP.get(key, {}))
    if any(key.startswith(p) for p in MANAGEMENT_PREFIXES):
        d["management"] = True
    return d


def is_management(label: str) -> bool:
    return bool(label_dimensions(label).get("management"))


# --------------------------------------------------------------------------
# 文本规则（论坛标题/正文兜底；也用于 Issue 无对应标签时的补充信号）
# --------------------------------------------------------------------------
TEXT_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "entry": [
        ("app-server", r"\bapp[-\s]?server\b"),
        ("tui", r"\btui\b|\bterminal ui\b"),
        ("cli", r"\bcli\b|\bcommand[-\s]?line\b|\bcodex exec\b|\bterminal\b"),
        ("ios", r"\bios\b|\biphone\b|\bipad\b|\bandroid\b|\bmobile(?:\s+app)?\b"),
        ("extension", r"\bextension\b|\bvs ?code\b|\bjetbrains\b|\bplugin\b"),
        ("sdk", r"\bsdk\b|\bgithub action\b|\bapi key\b"),
        ("web", r"\bcodex[-\s]?web\b|\bweb app\b|\bweb ui\b|\bwebsite\b|\bdashboard\b"),
        # 仅将明确的 app/desktop 表述视为入口；品牌词和界面词不能单独
        # 判断用户使用了哪个入口。界面词仍由 cap=ui 捕捉。
        ("app", r"\bapp\b|\bdesktop\b"),
    ],
    "cap": [
        ("mcp", r"\bmcp\b|\bconnector\b"),
        ("remote", r"\bremote\b|\bssh\b|\bcloud agent\b|\bremote control\b"),
        ("sandbox", r"\bsandbox\b|\bpermission\b|\bapproval\b|\baccess control\b"),
        ("session", r"\bsession\b|\bthread\b|\bconversation\b|\bhistory\b|\bresume\b|\bcheckpoint\b|\bcontext window\b"),
        ("tool-calls", r"\btool call|\bshell\b|\bexec\b|\bpatch\b|\bterminal\b"),
        ("models", r"\bmodel\b|\bgpt\b|\bclaude\b|\bsonnet\b|\bbyok\b|\bmax mode\b"),
        ("usage", r"\busage\b|\bbilling\b|\bsubscription\b|\brate limit|\bquota\b|\bpricing\b|\bplan\b"),
        ("auth", r"\blogin\b|\bauth\b|\boauth\b|\baccount\b|\bsign in\b"),
        ("connectivity", r"\bconnect(?:ion|ivity)?\b|\bnetwork\b|\bproxy\b|\btimeout\b|\boffline\b"),
        ("agent", r"\bagent\b|\bsubagent\b|\bcomposer\b|\bplan mode\b|\bautomation\b"),
        ("code-review", r"\bcode review\b|\bbugbot\b"),
        ("ui", r"\bui\b|\blayout\b|\bsidebar\b|\beditor\b|\btheme\b|\bchat window\b|\bpanel\b"),
        ("config", r"\bconfig\b|\bsettings\b|\bkeybind|\brules?\b|\bhooks?\b"),
        ("workspace", r"\bworktree|\bgit\b|\bdiff\b|\bworkspace\b"),
        ("bot", r"\bbot\b|\bslack\b|\blinear\b"),
        ("deployment", r"\bazure\b|\bbedrock\b|\benterprise\b|\bon[-\s]?prem"),
    ],
    "env": [
        ("windows", r"\bwindows\b|\bwin(?:10|11)\b|\bwsl\b"),
        ("macos", r"\bmacos\b|\bmac os\b|\bmacbook\b|\bapple silicon\b"),
        ("linux", r"\blinux\b|\bubuntu\b|\bdebian\b|\bfedora\b"),
        ("browser", r"\bbrowser\b|\bchrome\b|\bsafari\b|\bfirefox\b"),
        ("mobile", r"\biphone\b|\bipad\b|\bandroid\b|\bios\b"),
    ],
    "type": [
        ("bug", r"\bcrash|\berror\b|\bfail(?:s|ed|ing)?\b|\bbroken\b|\bnot working\b|\bdoesn'?t work\b|\bbug\b|\bexception\b|\bfreez|\bhang(?:s|ing)?\b|\bregression\b|\bstuck\b|\bwrong\b"),
        ("performance", r"\bslow\b|\bperformance\b|\blatency\b|\bcpu\b|\bmemory usage\b|\blag(?:s|ging)?\b"),
        ("connectivity", r"\bconnect(?:ion|ivity)?\b|\bnetwork\b|\bproxy\b|\btimeout\b|\boffline\b|\bunable to connect\b"),
        ("help", r"\bhow (?:do|can|to)\b|\banyone\b|\bhelp\b|\bquestion\b|\bwhy does\b|\bconfused\b"),
        ("enhancement", r"\bfeature request\b|\bwould be (?:great|nice|helpful)\b|\bplease add\b|\bsuggest(?:ion)?\b|\bsupport for\b|\benhancement\b|\brequest(?:ing)?\b"),
        ("feedback", r"\bfeedback\b|\bux\b|\bconfusing\b|\bannoying\b|\bimprove(?:ment)?\b|\bfrustrat"),
        ("docs", r"\bdocumentation\b|\bdocs\b|\bguide\b|\btutorial\b"),
    ],
}
_COMPILED = {dim: [(code, re.compile(rx, re.I)) for code, rx in rules]
             for dim, rules in TEXT_PATTERNS.items()}

# 文本主类优先级：越靠前越优先
TEXT_TYPE_PRIORITY = ["bug", "performance", "connectivity", "help", "enhancement", "feedback", "docs"]


def classify_text(text: str) -> dict:
    """返回 {'entry': [...], 'cap': [...], 'env': [...], 'type': [...]}，均按优先级排序。"""
    text = text or ""
    out = {}
    for dim, rules in _COMPILED.items():
        hits = [code for code, rx in rules if rx.search(text)]
        order = ENTRY_ORDER if dim == "entry" else CAP_ORDER if dim == "cap" else ENV_ORDER if dim == "env" else TYPE_ORDER
        hits = sorted(set(hits), key=lambda x: order.index(x) if x in order else 99)
        out[dim] = hits
    return out


def label_signals(labels: Iterable[str]) -> dict:
    """把标签集合映射到各维度，返回集合与来源。"""
    entry, cap, env, typ = set(), set(), set(), set()
    management = False
    for lab in labels or []:
        d = label_dimensions(lab)
        entry.update(d.get("entry", []))
        cap.update(d.get("cap", []))
        env.update(d.get("env", []))
        typ.update(d.get("type", []))
        management = management or bool(d.get("management"))
    return {"entry": entry, "cap": cap, "env": env, "type": typ, "management": management}


def _ordered(values: Iterable[str], order: list[str]) -> list[str]:
    return sorted(set(values), key=lambda x: order.index(x) if x in order else 99)


def merge_signals(label_sig: dict, text_sig: dict, prefer_labels: bool = True) -> dict:
    """合并标签与文本信号。标签优先，但文本可补充缺失维度。"""
    out = {}
    for dim in ["entry", "cap", "env", "type"]:
        order = {"entry": ENTRY_ORDER, "cap": CAP_ORDER, "env": ENV_ORDER, "type": TYPE_ORDER}[dim]
        lv = list(label_sig.get(dim, []))
        tv = list(text_sig.get(dim, []))
        if prefer_labels and lv:
            vals = lv + [x for x in tv if x not in lv]
        else:
            vals = lv + [x for x in tv if x not in lv]
        out[dim] = _ordered(vals, order)
    out["management"] = bool(label_sig.get("management"))
    return out


def primary(values: list[str], order: list[str], fallback: str = "unknown") -> str:
    for v in order:
        if v in values:
            return v
    return fallback


def outcome_from_issue(state: str, reason: str | None) -> str:
    # GitHub 的 reopened 记录通常 state=open 且 state_reason=reopened；
    # 必须先看 reason，否则 154 条重开记录会被吞进普通 open。
    r = (reason or "").strip().lower()
    if r == "reopened":
        return "reopened"
    if state == "open":
        return "open"
    if r == "completed":
        return "completed"
    if r == "duplicate":
        return "duplicate"
    if r == "not_planned":
        return "not_planned"
    return "closed_unspecified"


def taxonomy_markdown() -> str:
    lines = ["# 统一分类字典（入口—能力—环境—问题类型—处理结果）", "",
             "> 本文件由 `src/common/taxonomy.py` 自动生成，请不要手工修改；改字典请改代码。", "",
             "## 使用规则", "",
             "1. 每个维度允许多标签；`*_primary` 是报告分布时使用的主类。",
             "2. 标签（Issue labels / 论坛 tags）优先于文本规则；文本只补充缺失维度。",
             "3. 管理型标签（Papercuts、affected:、source: 等）保留在原始表，但不进入产品问题结论。",
             "4. 维度无法判定时保留 `unknown`，不强行归类。",
             "5. 关联规则只表示标签共现，不表示因果或真实故障发生率。", ""]
    for title, defs in [("产品入口", ENTRY_DEF), ("能力模块", CAP_DEF), ("运行环境", ENV_DEF),
                        ("问题类型", TYPE_DEF), ("处理结果", OUTCOME_DEF)]:
        lines += [f"## {title}", "", "| 代码 | 中文名 | 判定说明 |", "|---|---|---|"]
        for k, (zh, desc) in defs.items():
            lines.append(f"| `{k}` | {zh} | {desc} |")
        lines.append("")
    lines += ["## 标签映射（Issue labels / 论坛 tags）", "",
              "| 标签 | 入口 | 能力 | 环境 | 问题类型 | 管理型 |", "|---|---|---|---|---|---|"]
    for name in sorted(LABEL_MAP):
        d = LABEL_MAP[name]
        mg = "是" if d.get("management") else ""
        lines.append("| `%s` | %s | %s | %s | %s | %s |" % (
            name, ", ".join(d.get("entry", [])), ", ".join(d.get("cap", [])),
            ", ".join(d.get("env", [])), ", ".join(d.get("type", [])), mg))
    lines += ["", "## 竞争风险编码（RQ3 扩展，2026-10-23）", "",
              "关闭原因在生存分析中作为互斥竞争事件；与 `OUTCOME_DEF` 的映射如下：", "",
              "| 事件码 | 含义 | 来源 | 说明 |", "|---|---|---|---|",
              "| 0 | 右删失 | 截点时仍 open / 截点后才关闭 | 仅表示“未观察到该原因下的关闭”，不代表未修复 |",
              "| 1 | completed | state=closed 且 state_reason=completed | 仍不等于已修复 |",
              "| 2 | duplicate | state=closed 且 state_reason=duplicate | 重复合并，不计入“完成” |",
              "| 3 | not_planned | state=closed 且 state_reason=not_planned | 主动不处理 |",
              "",
              "`reopened` 的早先关闭时间不可见：在竞争风险分析中只能按当前状态右删失（或整体排除），",
              "两种处理都在 `src/09_rq3_competing_risk.py` 中报告，不可与 `open` 混为一类。", ""]
    lines += ["", "## 文本规则", "",
              "| 维度 | 代码 | 正则（节选） |", "|---|---|---|"]
    for dim in ["entry", "cap", "env", "type"]:
        for code, rx in TEXT_PATTERNS[dim]:
            lines.append(f"| {dim} | `{code}` | `{rx}` |")
    return "\n".join(lines) + "\n"
