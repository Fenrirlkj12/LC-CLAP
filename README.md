# LC-CLAP Method Pattern

This repository explains the control flow of an LC-CLAP-style audio-text contrastive system without disclosing its private implementation. It is a method pattern, not an exact reproduction of a private training system.

```text
JSONL pairs + illustrative Leq proxy -> condition bucket -> conditioned text
                                    -> audio encoder + text encoder
                                    -> normalized embedding similarity
                                    -> context-aware hard negatives
                                    -> optimizer/checkpoint -> five-seed evaluation
```

The implementation shows the logical roles of Leq-conditioned prompts, context-aware hard negatives, independent seed evaluation, and aggregate result reporting alongside the audio encoder, text encoder, normalized embeddings, and contrastive objective.

All rules in [profiles/lc_clap_pattern.example.json](profiles/lc_clap_pattern.example.json) are illustrative stand-ins. Its bucket boundaries, prompt text, hard-negative scale, and five seed identifiers are not production values.

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

## Data Manifest

Training data is a UTF-8 JSONL file. Each line describes one paired audio/text sample. Relative audio paths are resolved relative to the manifest file.

```json
{"audio_path": "audio/0001.wav", "text": "fallback description", "label": "example_event", "leq_value": 0.42}
{"audio_path": "audio/0002.wav", "text": "fallback description", "label": "another_event", "leq_value": 0.81}
```

`label` and `leq_value` are optional. When both are present, the method profile maps the scalar proxy into an illustrative condition bucket and creates the paired text with its illustrative prompt policy. The scalar is a placeholder interface, not a disclosed Leq scale or threshold scheme.

## Training

```bash
python train.py \
    --manifest data/train.jsonl \
    --output-dir checkpoints/reference_run \
    --method-profile profiles/lc_clap_pattern.example.json \
    --seed 101 \
    --epochs <example_epoch_count> \
    --batch-size <example_batch_size> \
    --learning-rate <example_learning_rate>
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
## Five-Seed Evaluation and Results
## Scope and Results
Run each independently seeded training job with an illustrative seed identifier, then evaluate its checkpoint:

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

The aggregate tool calculates mean and population standard deviation across exactly five reports. Publish only reviewed aggregate values through [results/FINAL_RESULTS_TEMPLATE.md](results/FINAL_RESULTS_TEMPLATE.md), not per-seed outputs or private analysis artifacts.

For an end-to-end view of the control flow, [run_five_seed_protocol.py](run_five_seed_protocol.py) runs the five illustrative seed identifiers in the method profile, trains one run per identifier, evaluates each final checkpoint, writes internal per-run reports, and creates one aggregate report. It requires the same public-style inputs as the separate commands and should only be used with licensed data.

## Disclosure Boundary
Read [docs/PUBLICATION_SCOPE.md](docs/PUBLICATION_SCOPE.md) before extending the repository. It lists excluded system details, including Leq processing, prompt policies, hard-negative logic, and private evaluation material.
Read [docs/PUBLICATION_SCOPE.md](docs/PUBLICATION_SCOPE.md) before extending the repository. It distinguishes the abstract LC-CLAP method flow included here from real boundaries, prompts, weights, data, architecture choices, calibration, and experimental assets that remain private.
No quantitative results are included. [docs/RESULTS.md](docs/RESULTS.md) defines the allowed public record and includes a release-note template that does not disclose private results.

This project is available under the [MIT License](LICENSE).
