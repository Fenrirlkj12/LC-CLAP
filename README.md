# LC-CLAP: Leq-Conditioned Audio-Text Contrastive Learning

LC-CLAP is an audio-text contrastive learning framework for condition-aware audio understanding. This repository contains the public training and evaluation workflow for Leq-conditioned modeling, including prompt construction, hard-negative weighting, checkpointing, and five-seed reporting.

```text
JSONL pairs + Leq descriptor -> condition bucket -> conditioned text
                              -> audio encoder + text encoder
                              -> normalized embedding similarity
                              -> context-aware hard negatives
                              -> optimizer/checkpoint -> five-seed evaluation
```

The framework combines Leq-conditioned prompts, context-aware hard negatives, independent seed evaluation, and aggregate result reporting with audio and text encoders, normalized embeddings, and a symmetric contrastive objective.

The default configuration is defined in [profiles/lc_clap_default.json](profiles/lc_clap_default.json). It provides a complete public configuration for the framework; deployment-specific assets and operating settings are managed outside this repository.

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## Data Manifest

Training data is a UTF-8 JSONL file. Each line describes one paired audio/text sample. Relative audio paths are resolved relative to the manifest file.

```json
{"audio_path": "audio/rain_001.wav", "text": "steady rain on a roof", "label": "rain", "leq_value": 0.28}
{"audio_path": "audio/traffic_014.wav", "text": "road traffic near an intersection", "label": "traffic", "leq_value": 0.76}
```

`label` and `leq_value` are optional. When both are present, the LC-CLAP configuration maps the scalar descriptor to a condition bucket and builds a condition-aware training text. The loader converts multi-channel audio to mono, resamples it to the configured sample rate, and crops or zero-pads each record to the configured duration.

## Training

```bash
python train.py \
    --manifest data/train.jsonl \
    --output-dir checkpoints/baseline \
    --method-profile profiles/lc_clap_default.json \
    --seed 101 \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 1e-4
```

Every checkpoint stores the model configuration, model state, optimizer state, completed epoch, and global step. Resume an interrupted run with:

```bash
python train.py --manifest data/train.jsonl --resume checkpoints/baseline/last.pt
```

## Retrieval

```bash
python retrieve.py \
    --checkpoint checkpoints/baseline/last.pt \
    --audio input.wav \
    --labels rain traffic speech music \
    --top-k 3
```
## Five-Seed Evaluation

Run each independently seeded training job, then evaluate its checkpoint:

```bash
python evaluate.py \
    --checkpoint checkpoints/run_101/last.pt \
    --manifest data/evaluation.jsonl \
    --labels data/public_labels.json \
    --seed 101 \
    --output outputs/evaluation_101.json

python aggregate_results.py \
    --reports outputs/evaluation_101.json outputs/evaluation_203.json \
                        outputs/evaluation_307.json outputs/evaluation_401.json \
                        outputs/evaluation_509.json \
    --output outputs/five_seed_aggregate.json
```

The aggregate tool calculates the mean and population standard deviation across exactly five reports. Publication status and approved aggregate outcomes are recorded in [results/RESULTS.md](results/RESULTS.md).

For an end-to-end workflow, [run_five_seed_protocol.py](run_five_seed_protocol.py) runs the five seed identifiers in the default configuration, trains one run per seed, evaluates each final checkpoint, and writes an aggregate report. It requires the same inputs as the separate commands and should be used with licensed data.

## Release Scope

[docs/PUBLICATION_SCOPE.md](docs/PUBLICATION_SCOPE.md) describes the public framework components and the implementation-specific assets maintained outside this repository. [docs/RESULTS.md](docs/RESULTS.md) documents the reporting protocol.

This project is available under the [MIT License](LICENSE).
