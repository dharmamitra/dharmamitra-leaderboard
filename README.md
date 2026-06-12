# Dharmamitra Machine Translation Performance Leaderboard

Live page: https://dharmamitra.github.io/dharmamitra-leaderboard/

Translation quality of frontier and open models on classical Asian languages →
English (Tibetan, Sanskrit, Chinese), scored with BLEU, chrF, and BLEURT.
GEMBA is excluded while the upstream LLM-judge endpoint is unavailable.

`index.html` is generated from the evaluation results
(`evaluation/outputs/all_scores.csv` in the `mitra-rag-mt` repo) by
`build_leaderboard.py`.

In the leaderboard tables, BLEURT is **displayed as a relative 0–100 score**
(min–max normalized across the listed models, 100 = best / 0 = worst) so the
tightly-clustered raw values (~0.47–0.60) are easier to read. The rescaling is
display-only — raw BLEURT-20 stays in the cell `data-sort` attributes and the
embedded `DETAIL` data, and is shown raw in the per-language heatmap and each
model's row detail. Note: this is done in `index.html`; if `build_leaderboard.py`
regenerates the page it must reapply this normalization or the change is lost.
