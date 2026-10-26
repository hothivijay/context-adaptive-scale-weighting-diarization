# Table 8. Computational profile

Copied from the paper. Not a reproduced result.

| Module | Parameters | Updated in training | Cost in T | Notes |
|---|---|---|---|---|
| MarbleNet voice activity detection | 0.31 M | no | linear | frozen, pre-trained |
| TitaNet-Large embedding, five branches | 23.0 M | no | linear, S passes | dominates latency, parallelizable |
| NME-SC clustering | --- | no | cubic in segments per session | session-level, blocks streaming |
| Reliability scorer + recurrent decoder | ~1.5 M | yes | linear | ~6% of parameters, no attention |
| Complete system | ~24.81 M | ~6% | linear except clustering | 0.045 real-time factor, one GTX 1080 |
