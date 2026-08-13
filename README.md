# CLAP Reference Implementation

This repository provides a compact audio-text contrastive learning baseline. It contains a complete, general-purpose training path rather than a fixed inference stub:

```text
JSONL pairs -> WAV loading and resampling -> audio encoder + text encoder
                     -> normalized projection embeddings -> symmetric contrastive loss
                     -> optimizer update -> resumable checkpoints -> text retrieval
```

The implementation is intentionally conservative: it uses a log-Mel convolutional audio encoder, a Transformer text encoder, normalized embedding projections, a learned temperature, and a symmetric InfoNCE loss. It is suitable as a readable baseline or an integration foundation.

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## Data Manifest

Training data is a UTF-8 JSONL file. Each line describes one paired audio/text sample. Relative audio paths are resolved relative to the manifest file.

```json
{"audio_path": "audio/0001.wav", "text": "rain falling on a metal roof"}
{"audio_path": "audio/0002.wav", "text": "a commuter train arriving at a platform"}
```

The loader converts multi-channel audio to mono, resamples it to the configured sample rate, and crops or zero-pads each record to the configured duration.

## Training

```bash
python train.py \
    --manifest data/train.jsonl \
    --output-dir checkpoints/reference_run \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 1e-4
```

Every checkpoint stores the model configuration, model state, optimizer state, completed epoch, and global step. Resume an interrupted run with:

```bash
python train.py --manifest data/train.jsonl --resume checkpoints/reference_run/last.pt
```

## Retrieval

```bash
python retrieve.py \
    --checkpoint checkpoints/reference_run/last.pt \
    --audio input.wav \
    --labels rain traffic speech music \
    --top-k 3
```

## Public-Scope Guidance

This is a general baseline, not a disclosure of a production system. Keep trained weights, source audio, manifests, experiments, secrets, internal endpoints, custom preprocessing, curated vocabularies, prompts, and calibration policies out of the repository. Review Git history before publication as well as the current working tree.
