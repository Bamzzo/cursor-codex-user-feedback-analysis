# -*- coding: utf-8 -*-
"""通用统计与输出工具（不依赖 statsmodels / lifelines）。"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy import stats


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def km(durations, events):
    """Kaplan-Meier；返回逐时点表（含 95% log-log 置信区间）。"""
    z = pd.DataFrame({"t": np.asarray(durations, float), "e": np.asarray(events, int)})
    z = z.dropna()
    tab = z.groupby("t").agg(events=("e", "sum"), removed=("e", "size")).sort_index()
    n = len(z)
    s, g, rows = 1.0, 0.0, []
    for t, r in tab.iterrows():
        e = int(r.events)
        if e:
            s *= 1 - e / n
            if n > e:
                g += e / (n * (n - e))
        if 0 < s < 1:
            se = np.sqrt(g) / abs(np.log(s))
            lo = np.exp(-np.exp(np.log(-np.log(s)) + 1.96 * se))
            hi = np.exp(-np.exp(np.log(-np.log(s)) - 1.96 * se))
        else:
            lo = hi = s
        rows.append([t, n, e, int(r.removed) - e, s, lo, hi])
        n -= int(r.removed)
    return pd.DataFrame(rows, columns=["days", "at_risk", "events", "censored",
                                       "survival", "lower95", "upper95"])


def km_value(curve: pd.DataFrame, day: float, col: str = "survival") -> float:
    x = curve[curve.days <= day]
    return float(x.iloc[-1][col]) if len(x) else 1.0


def km_median(curve: pd.DataFrame) -> float | None:
    x = curve[curve.survival <= 0.5]
    return float(x.iloc[0].days) if len(x) else None


def logrank(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """多组 log-rank 检验。groups=[(time, event), ...]。"""
    allt = np.unique(np.concatenate([g[0] for g in groups]))
    obs = np.zeros(len(groups))
    exp = np.zeros(len(groups))
    var = np.zeros((len(groups), len(groups)))
    for t in allt:
        atrisk, d, n = [], [], 0
        for time, event in groups:
            r = (time >= t).sum()
            e = ((time == t) & (event == 1)).sum()
            atrisk.append(r)
            d.append(e)
            n += r
        dsum = sum(d)
        if n <= 1 or dsum == 0:
            continue
        for i in range(len(groups)):
            obs[i] += d[i]
            exp[i] += dsum * atrisk[i] / n
        if n > 1:
            for i in range(len(groups)):
                for j in range(len(groups)):
                    if i == j:
                        var[i, i] += dsum * (atrisk[i] / n) * (1 - atrisk[i] / n) * (n - dsum) / (n - 1)
                    else:
                        var[i, j] -= dsum * (atrisk[i] / n) * (atrisk[j] / n) * (n - dsum) / (n - 1)
    try:
        inv = np.linalg.pinv(var)
        chi2 = float((obs - exp) @ inv @ (obs - exp))
        df = len(groups) - 1
        return chi2, float(stats.chi2.sf(chi2, df))
    except Exception:
        return np.nan, np.nan


def cramers_v(table: pd.DataFrame) -> tuple[float, float, int]:
    arr = table.to_numpy()
    chi2, p, dof, _ = stats.chi2_contingency(arr)
    n = arr.sum()
    v = np.sqrt(chi2 / (n * (min(arr.shape) - 1))) if n > 0 and min(arr.shape) > 1 else np.nan
    return float(chi2), float(p), float(v)


def bootstrap_metric(y_true, y_score, fn, n_boot: int = 300, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[ix])) < 2:
            continue
        vals.append(fn(y_true[ix], y_score[ix]))
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]))


def pct(x: float, digits: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def md_table(df: pd.DataFrame, max_rows: int | None = None, floatfmt: str = ".3f") -> str:
    d = df.head(max_rows) if max_rows else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append("—" if pd.isna(v) else format(v, floatfmt))
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_csv(df: pd.DataFrame, path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_jsonable)
