# Dharmamitra Machine Translation Performance Leaderboard

Live page: https://dharmamitra.github.io/dharmamitra-leaderboard/

Translation quality of frontier and open models on classical Asian languages →
English (Tibetan, Sanskrit, Chinese), scored with BLEU, chrF, and BLEURT.
GEMBA is excluded while the upstream LLM-judge endpoint is unavailable.

`index.html` is generated from the evaluation results
(`evaluation/outputs/all_scores.csv` plus `external_scores.csv` for models
evaluated off-machine, both in the `mitra-rag-mt` repo) by
`build_leaderboard.py` (in this repo). The builder is template-preserving:
it replaces only the table bodies, the `DETAIL` data blob, the model count,
and `summary_overview.png` — page chrome and the display-side BLEURT
normalization JS survive regeneration. Runs named `*-s500` (sampled subsets)
and models missing any language pair are excluded automatically.

In the leaderboard tables, BLEURT is **displayed as a relative 0–100 score**
(min–max normalized across the listed models, 100 = best / 0 = worst) so the
tightly-clustered raw values (~0.47–0.60) are easier to read. The rescaling is
display-only — raw BLEURT-20 stays in the cell `data-sort` attributes and the
embedded `DETAIL` data, and is shown raw in the per-language heatmap and each
model's row detail. The normalization lives in the page's JS, which the
template-preserving builder leaves untouched across regenerations.
