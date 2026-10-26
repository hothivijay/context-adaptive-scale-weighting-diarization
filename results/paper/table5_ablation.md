# Table 5. Cumulative component-wise ablation

Copied from the paper. Not a reproduced result.

| Configuration | Dev (%) | Eval (%) | Delta eval vs previous row (pp) |
|---|---|---|---|
| Baseline (DISPLACE-2024) [43] * | 25.45 | 29.96 | --- |
| + MarbleNet VAD [37] | 23.30 | 27.21 | -2.75 |
| + TitaNet-Large embeddings [38] | 24.60 | 26.75 | -0.46 |
| + Multi-scale decoder, static weights [15] | 23.43 | 25.76 | -0.99 |
| + Dynamic scale weighting (full system) | 19.35 | 21.46 | -4.30 |

* The baseline row reproduces figures reported in [43] on DISPLACE 2023 data and must be re-run on DISPLACE 2024 data. Dev DER for the rows that include the trained decoder is training-set error.
