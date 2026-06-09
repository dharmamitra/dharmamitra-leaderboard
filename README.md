# Dharmamitra Machine Translation Performance Leaderboard

Live page: https://dharmamitra.github.io/dharmamitra-leaderboard/

Translation quality of frontier and open models on classical Asian languages →
English (Tibetan, Sanskrit, Chinese), scored with BLEU, chrF, and BLEURT.
GEMBA is excluded while the upstream LLM-judge endpoint is unavailable.

`index.html` is generated from the evaluation results
(`evaluation/outputs/all_scores.csv` in the `mitra-rag-mt` repo) by
`build_leaderboard.py`.
