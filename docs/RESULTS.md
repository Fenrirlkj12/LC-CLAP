# Results Policy

The repository may publish reviewed final results that explain the public LC-CLAP method pattern. Results must be aggregate-only and must not disclose the underlying private configuration or detailed experiment traces.

## Allowed Final Record

Use [results/FINAL_RESULTS_TEMPLATE.md](../results/FINAL_RESULTS_TEMPLATE.md) for an approved release report. It may include:

- A high-level licensed dataset description.
- The statement that five independently seeded runs were aggregated.
- An approved aggregate metric and its dispersion.
- A non-comparative conclusion limited to the public release scope.

## Material That Stays Private

- Per-seed values, training curves, checkpoints, and raw predictions.
- Per-class results, confusion matrices, threshold analyses, and diagnostic plots.
- Exact prompt text, Leq thresholds, hard-negative settings, model configuration, and data-preparation details.
- Comparative claims about a private production system.
