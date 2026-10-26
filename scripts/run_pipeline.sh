#!/usr/bin/env bash
# Full run on DISPLACE 2024, steps 0-10. Edit the paths below first.
# Every step can also be run on its own; see README.md.
set -euo pipefail

DATA=${DATA:-data/displace2024}     # corpus location (docs/02_dataset.md)
NOISE=${NOISE:-}                    # optional: folder of 16 kHz noise files, e.g. MUSAN/noise
SEEDS=${SEEDS:-"42"}                # e.g. SEEDS="42 43 44" for the multi-seed part of Table 9
WORK=data/work
OUT=results/reproduced

python scripts/00_check_environment.py

# 1. manifests, 16 kHz check, measured Table 3
python scripts/01_prepare_data.py \
  --dev_audio  $DATA/dev/wav  --dev_rttm  $DATA/dev/rttm \
  --eval_audio $DATA/eval/wav --eval_rttm $DATA/eval/rttm --out $WORK/manifests

# 2. augmented copies of the development audio
python scripts/02_augment.py --dev_manifest $WORK/manifests/dev.json ${NOISE:+--noise_dir $NOISE}

# 3. frozen front end: oracle VAD for training data, MarbleNet for evaluation
python scripts/03_extract_embeddings.py --manifest $WORK/manifests/dev.json     --vad oracle    --out_dir $WORK/cache/dev --no_cluster
python scripts/03_extract_embeddings.py --manifest $WORK/manifests/dev_aug.json --vad oracle    --out_dir $WORK/cache/dev_aug --no_cluster
python scripts/03_extract_embeddings.py --manifest $WORK/manifests/eval.json    --vad marblenet --out_dir $WORK/cache/eval

# 5 (clustering only): the "+ TitaNet-Large" row of Table 5
python scripts/05_diarize.py --cache_dir $WORK/cache/eval --clustering_only --out_dir exp/clustering/eval
python scripts/06_score.py --ref_dir $DATA/eval/rttm --hyp_dir exp/clustering/eval/pred_rttms --out $OUT/eval_clustering

# 4-6 for each seed and both weighting rules
STATIC=(); DYNAMIC=()
for SEED in $SEEDS; do
  for W in static dynamic; do
    EXP=exp/${W}_seed${SEED}
    python scripts/04_train.py --cache_dirs $WORK/cache/dev $WORK/cache/dev_aug --weighting $W --seed $SEED --exp_dir $EXP
    python scripts/05_diarize.py --cache_dir $WORK/cache/eval --model $EXP/model.pt --out_dir $EXP/eval
    python scripts/06_score.py --ref_dir $DATA/eval/rttm --hyp_dir $EXP/eval/pred_rttms --out $OUT/eval_${W}_seed${SEED} --pyannote
  done
  STATIC+=($OUT/eval_static_seed${SEED}_per_file.csv)
  DYNAMIC+=($OUT/eval_dynamic_seed${SEED}_per_file.csv)
done
FIRST=$(echo $SEEDS | awk '{print $1}')
python scripts/06_score.py --ref_dir $DATA/eval/rttm --hyp_dir exp/dynamic_seed${FIRST}/eval/pred_rttms \
  --out $OUT/short_utterances_dynamic_seed${FIRST} --short_utterances

# 7. Table 9 (Algorithm 2)
python scripts/07_significance.py --static "${STATIC[@]}" --dynamic "${DYNAMIC[@]}"

# 8. Tables 6 and 7
python scripts/08_conditions.py --manifest $WORK/manifests/eval.json \
  --dynamic $OUT/eval_dynamic_seed${FIRST}_per_file.csv --static $OUT/eval_static_seed${FIRST}_per_file.csv

# 9. Table 8
python scripts/09_profile.py --nemo --extract_timing $WORK/cache/eval/timing.jsonl \
  --decode_timing exp/dynamic_seed${FIRST}/eval/timing.jsonl

# 10. tables and figures, with measured weights of the first evaluation session
NPZ=$(ls exp/dynamic_seed${FIRST}/eval/weights/*.npz | head -1)
python scripts/10_make_figures.py --weights_npz "$NPZ"
