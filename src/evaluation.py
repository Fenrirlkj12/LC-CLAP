"""Condition-aware evaluation and five-seed result aggregation interfaces."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .data import HashTokenizer, collate_batch
from .lc_pattern import LCMethodProfile
from .model import CLAPModel


def evaluate_conditioned_retrieval(
    model: CLAPModel,
    dataset: torch.utils.data.Dataset[dict[str, object]],
    labels: Sequence[str],
    method_profile: LCMethodProfile,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate label retrieval with a condition-specific text prototype per sample."""
    if not labels:
        raise ValueError("At least one candidate label is required")
    label_to_index = {label: index for index, label in enumerate(labels)}
    if len(label_to_index) != len(labels):
        raise ValueError("Candidate labels must be unique")
    tokenizer = HashTokenizer(model.config.vocabulary_size, model.config.max_text_tokens)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_batch(batch, tokenizer))
    correct = 0
    total = 0
    by_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            label = batch["labels"][0]
            if label not in label_to_index:
                raise ValueError("Each evaluation record must have a label from the candidate set")
            bucket = batch["condition_buckets"][0]
            if bucket is None:
                prompts = list(labels)
                bucket_name = "unconditioned"
            else:
                prompts = [method_profile.prompts.build(candidate, bucket)[0] for candidate in labels]
                bucket_name = bucket
            tokens, attention_mask = tokenizer.batch_encode(prompts)
            audio_embedding = model.encode_audio(batch["waveforms"].to(device))
            text_embeddings = model.encode_text(tokens.to(device), attention_mask.to(device))
            prediction = int((audio_embedding @ text_embeddings.T).argmax(dim=-1).item())
            is_correct = int(prediction == label_to_index[label])
            correct += is_correct
            total += 1
            by_bucket[bucket_name][0] += is_correct
            by_bucket[bucket_name][1] += 1
    return {
        "sample_count": total,
        "top1_accuracy": correct / total if total else 0.0,
        "by_condition_bucket": {
            bucket: {"top1_accuracy": hits / count, "sample_count": count}
            for bucket, (hits, count) in sorted(by_bucket.items())
        },
    }


def aggregate_five_seed_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate five independently produced evaluation reports."""
    if len(reports) != 5:
        raise ValueError("Expected exactly five seed reports")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != 5:
        raise ValueError("Seed reports must have distinct seed identifiers")
    values = [float(report["metrics"]["top1_accuracy"]) for report in reports]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "seed_count": len(reports),
        "aggregate": {"top1_accuracy_mean": mean, "top1_accuracy_std": math.sqrt(variance)},
        "report_schema": "aggregate-only; do not publish seed-level inputs without review",
    }


def write_json(payload: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
