# Publication Scope

This repository is an LC-CLAP method pattern. It documents the public control flow of conditioned audio-text contrastive learning without disclosing the private implementation or research artifacts that instantiate that flow.

## Included

- An illustrative Leq-proxy to condition-bucket interface.
- Illustrative condition-aware prompt construction.
- Illustrative same-condition/different-class hard-negative weighting.
- Audio and text encoder interfaces, contrastive training, checkpointing, and retrieval.
- A five-seed evaluation and aggregate reporting interface.

The code is intentionally self-contained and uses illustrative defaults selected for readability. They are not derived from, calibrated against, or intended to describe the private system's actual values.

## Excluded

Do not add the following material to this repository:

- Production architectures, weights, checkpoints, tokenizer assets, or configuration files.
- Domain feature engineering, signal processing rules, calibration, or post-processing.
- Actual Leq scales, bucket thresholds, domain labels, generation logic, or associated metadata.
- Actual prompt templates, prompt-selection rules, hard-negative selection, mining policies, or curriculum logic.
- Private datasets, manifests, audio, annotations, experimental logs, or internal paths.
- Internal services, endpoints, credentials, environment files, or deployment scripts.
- Unreviewed per-seed outputs, class-level results, figures, or claims from private experiments.

## Extension Policy

Extensions should preserve the public interface while keeping private algorithms and tuned values outside this repository. Reviewed, aggregate-only final results may be published through [results/FINAL_RESULTS_TEMPLATE.md](../results/FINAL_RESULTS_TEMPLATE.md).
