# The DISPLACE 2024 dataset

## What it is

From the paper (section "DISPLACE 2024 Corpus" and Table 3):

- multilingual, multi-party conversations collected at four academic institutions,
  released as the successor of DISPLACE 2023
- development partition: 35 recordings, about 20 h
- evaluation partition: 32 recordings, about 18 h (about 38 h in total)
- sessions of 30 to 60 minutes, 16 kHz, far-field microphone arrays plus close-talk lapel references
- typically 60-75 % single-speaker speech, 20-35 % overlapped speech, 5-15 % non-speech;
  turns mostly 2-6 s; reverberation time 0.4-0.9 s; SNR 5-20 dB

The challenge provides **no training partition**. Like the paper, this code
trains on the development partition and evaluates on the evaluation partition.
The development DER of a trained model is therefore training-set error. Only the
evaluation DER is held out.

The paper gives typical ranges and flags one inconsistency itself: 3-8
speakers per session according to the annotation, against 3-5 in the
challenge description. Step 1 (`scripts/01_prepare_data.py`) measures all of
these on your copy and writes `results/reproduced/table3_corpus_measured.md`,
so you can quote measured values.

## Getting it

The data is distributed by the DISPLACE organisers for research, on request
and under their data-usage terms: **https://displace2024.github.io/** (the
address given in the paper's Data Availability Statement). The companion
PReMI 2025 paper also points to Zenodo record https://zenodo.org/records/12166687.
Read and accept the terms before downloading. This repository does not
redistribute any audio or labels.

Use the speaker diarization track (Track 1) labels.

## Arranging the files

The scripts only need two folders per partition, audio and RTTM, with matching
base names:

```
data/displace2024/
├── dev/
│   ├── wav/    <session>.wav ...
│   └── rttm/   <session>.rttm ...
└── eval/
    ├── wav/
    └── rttm/
```

Use whatever session IDs the release uses, but keep each audio file and its RTTM
under the same base name. If the release ships labels in another layout, one
symbolic link per file is enough.

## Audio format

Everything must be 16 kHz, mono wav (Table 2). Step 1 stops with a message if a
file is not. To convert a folder:

```bash
mkdir -p dev/wav16
for f in dev/wav/*.wav; do ffmpeg -loglevel error -i "$f" -ar 16000 -ac 1 "dev/wav16/$(basename "$f")"; done
```

If your copy has several audio streams per session, use the far-field stream the
challenge designates for the diarization track, and write down which one you
used. Close-talk (lapel) audio makes the task much easier, so a result on it is
not comparable with Table 4.

## Noise for augmentation (optional)

The paper adds noise at 10-20 dB SNR but does not name the source. MUSAN
(https://www.openslr.org/17/, the `noise/` part) is the usual choice. Point
`scripts/02_augment.py --noise_dir` at a folder of 16 kHz noise files. Without
it, the script falls back to white noise and says so.

## Disk space

Roughly: the corpus itself, plus one augmented copy of the development audio
(about the same size again), plus the feature caches of step 3 (five scales x
192 values per 0.25 s step, stored as float16; about 25-30 MB per hour of
speech).
