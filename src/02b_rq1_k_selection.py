# -*- coding: utf-8 -*-
"""RQ1 扩展：在固定预处理和随机种子下扫描候选主题数。

该扫描用于解释 K=8 的选择约束，不自动替换主模型，也不把单一指标称为真值。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DERIVED, OUT, FIG, SEED, setup_matplotlib
from common.stats import save_csv

OUTD = OUT / "rq1"
OUTD.mkdir(parents=True, exist_ok=True)
plt = setup_matplotlib()


def umass_coherence(H: np.ndarray, terms: list[str], token_sets: list[set[str]], topn: int = 10) -> float:
    vals = []
    for row in H:
        top = [terms[i] for i in np.argsort(row)[::-1][:topn]]
        df = {w: sum(w in s for s in token_sets) for w in top}
        for i in range(1, len(top)):
            for j in range(i):
                wi, wj = top[i], top[j]
                pair = sum(wi in s and wj in s for s in token_sets)
                vals.append(np.log((pair + 1) / (df[wj] + 1e-9)))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    docs = pd.read_parquet(DERIVED / "forum_docs.parquet")
    corpus = docs[docs["tok_text"].str.len() > 20].copy()
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.60, max_features=30000,
                                 ngram_range=(1, 1), sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w\w+\b")
    X = vectorizer.fit_transform(corpus["tok_text"])
    terms = vectorizer.get_feature_names_out().tolist()
    token_sets = [set(x.split()) for x in corpus["tok_text"]]
    norm_x_sq = float(X.multiply(X).sum())
    # 同一批文档用于每个候选 K，避免抽样差异影响轮廓分数。
    rng = np.random.default_rng(SEED)
    sample = np.sort(rng.choice(X.shape[0], size=min(1500, X.shape[0]), replace=False))
    rows = []
    for k in (4, 6, 8, 10, 12, 14):
        model = NMF(n_components=k, init="nndsvda", random_state=SEED,
                    max_iter=600, beta_loss="frobenius")
        W = model.fit_transform(X)
        H = model.components_
        labels = W.argmax(axis=1)
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
        sample_labels = labels[sample]
        sil = silhouette_score(Wn[sample], sample_labels, metric="cosine") if len(np.unique(sample_labels)) > 1 else np.nan
        top_lists = [set(np.argsort(row)[::-1][:10]) for row in H]
        diversity = len(set.union(*top_lists)) / (10 * k)
        rel_error = float(model.reconstruction_err_ ** 2 / norm_x_sq)
        rows.append({
            "k": k,
            "n_docs": int(X.shape[0]),
            "reconstruction_error": float(model.reconstruction_err_),
            "relative_reconstruction_error": rel_error,
            "reconstruction_share": 1.0 - rel_error,
            "umass_coherence_top10": umass_coherence(H, terms, token_sets),
            "topic_diversity_top10": float(diversity),
            "silhouette_cosine": float(sil),
            "silhouette_sample_n": int(len(sample)),
            "selected_for_main_analysis": int(k == 8),
            "seed": SEED,
            "selection_note": "K=8 retained for interpretability and continuity; scan is multi-criterion diagnostic, not automatic optimizer",
        })
        print(f"K={k}: rel_reconstruction={rel_error:.4f}, coherence={rows[-1]['umass_coherence_top10']:.4f}, silhouette={sil:.4f}", flush=True)
    result = pd.DataFrame(rows)
    save_csv(result, OUTD / "09_k_selection.csv")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2))
    panels = [
        ("reconstruction_share", "Relative reconstruction share", "Higher = closer reconstruction"),
        ("umass_coherence_top10", "UMass coherence", "Higher (less negative) = more co-occurring top terms"),
        ("silhouette_cosine", "Cosine silhouette", "Hard assignment on normalized document-topic weights"),
        ("topic_diversity_top10", "Top-10 topic diversity", "Unique top terms / total top-term slots"),
    ]
    for ax, (col, title, subtitle) in zip(axs.flat, panels):
        ax.plot(result.k, result[col], marker="o", color="#167D8D", linewidth=2)
        ax.axvline(8, color="#D47736", linestyle="--", linewidth=1.5, label="Current K=8")
        ax.set_title(title, fontsize=13, loc="left", fontweight="bold")
        ax.set_xlabel("Number of topics (K)")
        ax.set_ylabel(col.replace("_", " "))
        ax.grid(axis="y", alpha=.22)
        ax.legend(frameon=False, fontsize=9)
        ax.text(.02, .02, subtitle, transform=ax.transAxes, fontsize=8, color="#586B78")
    fig.suptitle("RQ1 topic-count diagnostics (fixed corpus, preprocessing and seed)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(FIG / "rq1_11_k选择诊断.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTD / '09_k_selection.csv'} and {FIG / 'rq1_11_k选择诊断.png'}")


if __name__ == "__main__":
    main()
