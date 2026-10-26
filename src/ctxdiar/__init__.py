"""Context-adaptive temporal scale weighting for far-field speaker diarization.

Reproduction code for V. Hothi and N. B. Gohil, "Context-adaptive temporal
scale weighting for overlap-aware far-field speaker diarization".

Module map (in pipeline order):
    vad.py         speech regions: oracle (training) or MarbleNet (inference)
    scales.py      five-scale windows and the common time index t
    frontend.py    frozen MarbleNet and TitaNet-Large (NVIDIA NeMo)
    nmesc.py       NME-SC clustering, Eqs. (1)-(2)
    features.py    one cached feature file per session
    model.py       reliability scorer + LSTM decoder, Eqs. (3)-(8), (10)
    data.py        training examples (chunk x speaker pair)
    decode.py      inference over all speaker pairs
    postprocess.py threshold, smoothing, RTTM
    der.py         DER, Eq. (11)
    stats.py       Algorithm 2: bootstrap CI, Wilcoxon, win rate
    conditions.py  corpus statistics and Table 6/7 breakdowns
    paper.py       every number printed in the paper
"""

__version__ = "1.0.0"
