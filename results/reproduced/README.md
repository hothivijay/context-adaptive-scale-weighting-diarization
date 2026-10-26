# Reproduced results

This folder is filled by the scripts when you run the pipeline on the real
DISPLACE 2024 data. In the repository it holds only this note and
`table5_reproduced.md`, which lists every row as pending: **no number appears
here until a run has actually produced it.**

| Run | Paper | Written by | Status |
|---|---|---|---|
| `table3_corpus_measured.*` | Table 3 (typical ranges) | step 1 | pending: needs the corpus |
| `eval_clustering.*` | Table 5, "+ TitaNet-Large": 26.75 | steps 5 (`--clustering_only`) and 6 | pending: needs a GPU run |
| `eval_static_seed<N>.*` | Table 5, static weights: 25.76 | steps 4-6 | pending: needs a GPU run |
| `eval_dynamic_seed<N>.*` | Tables 4, 5, full system: 21.46 | steps 4-6 | pending: needs a GPU run |
| `short_utterances_*.json` | Table 7, < 2 s: 26.53 | step 6 `--short_utterances` | pending |
| `table9_significance.*` | Table 9: not computed in the paper | step 7 | pending |
| `conditions*.{csv,md}` | Tables 6 and 7 | step 8 | pending |
| `table8_profile.*` | Table 8: 24.81 M, RTF 0.045 | step 9 | pending (parameter count works without data) |
| `table5_reproduced.md` | Table 5, rows 3-5 | step 10 | all pending |

Also look at `exp/<run>/eval/scale_weight_variance.csv` (V_s of Eq. 10). It
must be 0 for the static model and should be above 0 for the dynamic one.

When a run finishes, record underneath: commit hash, GPU, PyTorch and NeMo
versions, MarbleNet checkpoint, seed(s), real or white augmentation noise, and
whether `model.pt` (last epoch) or `model_best.pt` was used.

Files written per scoring run:

- `<name>.json`: corpus DER with MISS / FA / CONF and the scoring setting
- `<name>_per_file.csv`: the same per session (input to steps 7 and 8)
- `<name>.uem`: the scored regions (`--short_utterances` runs only)
