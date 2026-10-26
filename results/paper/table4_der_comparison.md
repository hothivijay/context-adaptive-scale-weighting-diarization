# Table 4. DER (%) on DISPLACE, zero collar

Copied from the paper. Not a reproduced result.

| System | Dev (%) | Eval (%) | Note in the paper |
|---|---|---|---|
| DISPLACE-2023 baseline, on DISPLACE 2023 data [43] | 27.33 | 32.18 | dagger: DISPLACE 2023 data, reference only |
| DISPLACE-2024 baseline, on DISPLACE 2023 data [43] | 25.45 | 29.96 | dagger: DISPLACE 2023 data, reference only |
| DISPLACE-2024 baseline, on DISPLACE 2024 data [22], [52] | 29.16 | 34.76 |  |
| pyannote.audio diarization 3.1 [31], [35] | 29.53 | 34.96 | source still to be confirmed by the authors |
| Hybrid diarization system [47] | 26.37 | 28.04 | source still to be confirmed by the authors |
| TCG CREST system [52] | 29.61 | 32.22 |  |
| TalTech-IRIT-LIS, PixIT + DPTNet, post-evaluation [22] | --- | 27.15 | post-evaluation submission |
| TalTech-IRIT-LIS, PixIT + DPTNet + 10 s sliding window, post-evaluation [22] | --- | 26.70 | post-evaluation submission |
| Proposed system | 19.35 | 21.46 | double dagger: dev DER is training-set error |

Rows marked 'DISPLACE 2023 data' are not comparable with the 2024 rows. The proposed system's dev DER is training-set error (the decoder is trained on the development partition).
