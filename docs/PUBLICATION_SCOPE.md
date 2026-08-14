# Framework Scope

This repository provides the LC-CLAP framework for Leq-conditioned audio-text contrastive learning. It includes the public training and evaluation workflow while keeping implementation-specific assets outside the source release.

## Included

- A Leq-compatible scalar-to-condition-bucket interface.
- Condition-aware prompt construction.
- Same-condition, different-class hard-negative weighting.
- Audio and text encoder interfaces, contrastive training, checkpointing, and retrieval.
- A five-seed evaluation and aggregate reporting interface.

The default profile provides a complete public configuration for training and evaluation. Production-specific values, assets, and operational settings are intentionally managed separately.

## Release Boundary

The following material is outside this source release:

- Production architectures, weights, checkpoints, tokenizer assets, or configuration files.
- Domain feature engineering, signal processing rules, calibration, or post-processing.
- Production Leq scales, bucket thresholds, domain labels, generation logic, or associated metadata.
- Production prompt templates, prompt-selection rules, hard-negative selection, mining policies, or curriculum logic.
- Private datasets, manifests, audio, annotations, experimental logs, or internal paths.
- Internal services, endpoints, credentials, environment files, or deployment scripts.
- Unreviewed per-seed outputs, class-level results, figures, or claims from private experiments.

## Extension Policy

Extensions should preserve the public interfaces and separate proprietary algorithms and tuned values from this repository. Approved aggregate metrics are recorded in [results/RESULTS.md](../results/RESULTS.md).
