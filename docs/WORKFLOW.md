# Workflow

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Manifest

Use UTF-8 JSONL with at least `audio_path` and `label` on each line. B3, B4, B5, and B7 require `leq_value`; the optional `split` field enables within-split shuffling.

```json
{"audio_path":"audio/clip_001.wav","label":"Light vehicle pass-by","leq_value":69.4,"split":"train"}
```

Audio is converted to mono, resampled to 48 kHz, and cropped or padded to 10 seconds.

## Training

```bash
python train.py \
  --manifest data/train.jsonl \
  --task-config configs/roadside_events_public.json \
  --experiment B4 \
  --pretrained-audio laion/clap-htsat-unfused \
  --pretrained-text roberta-base \
  --output-dir runs/B4_seed3408
```

B0 and B1 use audio classification heads. B2 performs audio-text contrastive learning without Leq prompts. B3 and B4 use the same hard Leq bucket prompts. B4 additionally weights negative samples from different classes within the same Leq bucket and applies SpecAugment time-frequency masking to HTSAT input features; the related parameters are defined under `experiments.B4` in the task configuration. Default epochs, batch size, learning rate, weight decay, workers, and seed values are defined under `training_defaults` and can be overridden from the command line.

B0 accepts a compatible PANNs checkpoint through `--pretrained-audio`. B1 accepts a Hugging Face HTSAT model ID or a local Transformers directory. B2-B4 require both `--pretrained-audio` and `--pretrained-text`.

B5 and B7 are Leq-shuffling controls based on B3:

```bash
python train.py --manifest data/train.jsonl \
  --task-config configs/roadside_events_public.json \
  --experiment B5_shuffle_control \
  --pretrained-audio laion/clap-htsat-unfused \
  --pretrained-text roberta-base \
  --output-dir runs/B5_seed3408
```

## Evaluation

```bash
python evaluate.py \
  --checkpoint runs/B4_seed3408/last.pt \
  --manifest data/validation.jsonl \
  --task-config configs/roadside_events_public.json \
  --output outputs/B4_seed3408_validation.json
```

The output includes Top-1 accuracy, Macro-F1, and per-class metrics.

## Single-File Inference

```bash
python retrieve.py \
  --checkpoint runs/B4_seed3408/last.pt \
  --audio data/example.wav \
  --task-config configs/roadside_events_public.json \
  --leq 71.2 \
  --top-k 5
```

B3 and B4 inference require `--leq`; B2, B0, and B1 do not.
