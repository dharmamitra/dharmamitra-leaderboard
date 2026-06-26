#!/usr/bin/env python3
"""Regenerate the leaderboard page from mitra-rag-mt's all_scores.csv.

Template-preserving: index.html is read and only the generated regions are
replaced — the two <tbody> blocks, the `const DETAIL = {...};` line, the
model count in the subtitle, and summary_overview.png. Everything else
(styles, notes, the display-side BLEURT 0-100 normalization JS) is left
untouched, so the hand-tuned page chrome survives regeneration.

Inclusion rules:
  - a model needs all three language pairs (bo-en, sa-en, zh-en) scored
  - run names ending in "-s500" are excluded (sampled, not comparable)
  - names containing "tb1024" go to the high-thinking-budget table

Usage: python3 build_leaderboard.py [path/to/all_scores.csv]
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DEFAULT = os.path.expanduser("~/code/mitra-rag-mt/evaluation/outputs/all_scores.csv")
INDEX = os.path.join(HERE, "index.html")
HEATMAP = os.path.join(HERE, "summary_overview.png")

DATASETS = ["bo-en", "sa-en", "zh-en"]
METRICS = ["BLEU", "CHRF", "Average_BLEURT"]
LANG_COL = {"bo-en": "Tibetan", "sa-en": "Sanskrit", "zh-en": "Chinese"}

# Display-name overrides (existing convention: production pipeline = MITRA (model)).
DISPLAY = {
    "mitra-3-flash-preview": "MITRA (gemini-3-flash-preview)",
    "mitra-3.1-flash-lite": "MITRA (gemini-3.1-flash-lite)",
    "mitra-3.5-flash": "MITRA (gemini-3.5-flash)",
    # gemma-2-mitra knn-RAG finetunes (local) + GRPO RL post-training
    "gemma-2-mitra-knn-sft-ckpt10-temp02": "MITRA-knn SFT (gemma-2-9B, ckpt-10)",
    "gemma-2-mitra-knn-sft-ckpt25-temp02": "MITRA-knn SFT (gemma-2-9B, ckpt-25)",
    "gemma-2-mitra-knn-sft-ckpt80-temp02": "MITRA-knn SFT (gemma-2-9B, ckpt-80)",
    "grpo-ckpt70": "MITRA-knn RL (gemma-2-9B, SFT+GRPO)",
    # BM25 (Tibetan char-ngram) retrieval variants — same pipeline, ES BM25 over the
    # 13M index instead of labse-knn vectors (no embedding model needed).
    "grpo-ckpt70-bm25": "MITRA-BM25 RL (gemma-2-9B, SFT+GRPO)",
    "gemma-3-12b-it-bm25-rag-temp02": "gemma-3-12b-it (BM25-RAG)",
    "mitra-3.1-flash-lite-bm25": "MITRA-BM25 (gemini-3.1-flash-lite)",
    "hunyuan-mt-7b-direct": "Hunyuan-MT-7B (direct)",
    # Qwen3.5-122B-A10B-AWQ local runs (no-knn + knn-RAG, thinking disabled);
    # groups with the off-machine Qwen3.5-27B/35B-A3B/9B comparison entries.
    "qwen3.5-122b-a10b-vanilla-temp02": "Qwen3.5-122B-A10B",
    "qwen3.5-122b-a10b-knn-rag-temp02": "Qwen3.5-122B-A10B (knn-RAG)",
    "qwen3.5-4b-vanilla-temp02": "Qwen3.5-4B",
    "qwen3.5-4b-knn-rag-temp02": "Qwen3.5-4B (knn-RAG)",
    "qwen3-8b-vanilla-temp02": "Qwen3-8B",
    "qwen3-8b-knn-rag-temp02": "Qwen3-8B (knn-RAG)",
}

EXCLUDE_SUFFIXES = ("-s500",)


def load_scores(path):
    data = defaultdict(lambda: defaultdict(dict))
    # external_scores.csv (same dir) holds models evaluated off-machine whose
    # raw outputs are not in the repo; plot_eval_results.py regenerates
    # all_scores.csv from local outputs only and would otherwise drop them.
    paths = [path]
    ext = os.path.join(os.path.dirname(path), "external_scores.csv")
    if os.path.exists(ext):
        paths.append(ext)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data[row["model"]][row["dataset"]][row["metric_name"]] = float(row["metric_value"])
    models = {}
    for model, ds in data.items():
        if any(model.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        if not all(d in ds and all(m in ds[d] for m in METRICS) for d in DATASETS):
            continue
        langs = {d: {m: ds[d][m] for m in METRICS} for d in DATASETS}
        avg = {m: sum(langs[d][m] for d in DATASETS) / len(DATASETS) for m in METRICS}
        models[model] = {
            "display": DISPLAY.get(model, model),
            "overall": round(avg["Average_BLEURT"], 3),
            "group": "tb1024" if "tb1024" in model else "standard",
            "langs": langs,
            "avg": avg,
        }
    return models


def cell_color(value, lo, hi):
    """Per-column RdYlGn shading, matching the page's existing palette."""
    t = 0.5 if hi <= lo else (value - lo) / (hi - lo)
    r, g, b, _ = cm.get_cmap("RdYlGn")(0.05 + 0.90 * t)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def render_tbody(models, group):
    rows = [(name, m) for name, m in models.items() if m["group"] == group]
    rows.sort(key=lambda kv: -kv[1]["avg"]["Average_BLEURT"])
    if not rows:
        return ""
    ranges = {"overall": [min(m["avg"]["Average_BLEURT"] for _, m in rows),
                          max(m["avg"]["Average_BLEURT"] for _, m in rows)]}
    for d in DATASETS:
        vals = [m["langs"][d]["Average_BLEURT"] for _, m in rows]
        ranges[d] = [min(vals), max(vals)]

    out = []
    for rank, (name, m) in enumerate(rows, 1):
        o = m["avg"]["Average_BLEURT"]
        cells = (
            f'<td class="rank">{rank}</td>'
            f'<td class="model" data-text="{m["display"]}">{m["display"]}</td>'
            f'<td class="overall" data-sort="{o:.4f}" style="background:{cell_color(o, *ranges["overall"])}">{o:.3f}</td>'
        )
        for d in DATASETS:
            v = m["langs"][d]["Average_BLEURT"]
            cells += f'<td data-sort="{v:.4f}" style="background:{cell_color(v, *ranges[d])}">{v:.3f}</td>'
        out.append(f'<tr data-model="{name}">{cells}</tr>')
    return "\n".join(out)


def render_heatmap(models, path):
    rows = sorted(models.items(), key=lambda kv: -kv[1]["avg"]["Average_BLEURT"])
    names = [m["display"] for _, m in rows]
    cols = [(d, met) for d in DATASETS for met in METRICS]
    vals = np.array([[m["langs"][d][met] for d, met in cols] for _, m in rows])

    norm = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi <= lo else (col - lo) / (hi - lo)

    fig_h = 0.42 * len(rows) + 2.2
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{d}\n{('chrF' if met=='CHRF' else 'BLEURT' if met=='Average_BLEURT' else met)}"
                        for d, met in cols], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(vals.shape[0]):
        for j, (d, met) in enumerate(cols):
            v = vals[i, j]
            ax.text(j, i, f"{v:.3f}" if met == "Average_BLEURT" else f"{v:.1f}",
                    ha="center", va="center", fontsize=7)
    for x in (2.5, 5.5):
        ax.axvline(x, color="white", linewidth=2)
    ax.set_title("Raw scores per model, language pair and metric", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    models = load_scores(csv_path)
    html = open(INDEX, encoding="utf-8").read()

    n_std = sum(1 for m in models.values() if m["group"] == "standard")
    n_tb = len(models) - n_std

    # 1. model count in subtitle
    html = re.sub(r"Translation quality of \d+ models", f"Translation quality of {len(models)} models", html)

    # 2. main tbody (#board) and tb1024 tbody (#board-tb)
    bodies = [render_tbody(models, "standard"), render_tbody(models, "tb1024")]
    parts = re.split(r"(<tbody>\n).*?(\n  </tbody>)", html, flags=re.DOTALL)
    # re.split with 2 groups: [pre, open1, close1, mid, open2, close2, post]
    assert len(parts) == 7, f"expected 2 tbody blocks, found {(len(parts)-1)//3}"
    html = parts[0] + parts[1] + bodies[0] + parts[2] + parts[3] + parts[4] + bodies[1] + parts[5] + parts[6]

    # 3. DETAIL blob
    detail_json = json.dumps(models, ensure_ascii=False)
    html, n = re.subn(r"const DETAIL = \{.*?\};", f"const DETAIL = {detail_json};", html, flags=re.DOTALL)
    assert n == 1, "DETAIL blob not found"

    open(INDEX, "w", encoding="utf-8").write(html)
    render_heatmap(models, HEATMAP)
    print(f"index.html rebuilt: {n_std} standard + {n_tb} tb1024 models; heatmap updated")


if __name__ == "__main__":
    main()
