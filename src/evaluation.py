"""Task evaluation and metric reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from .data import HashTokenizer, PretrainedTokenizer
from .model import AudioOnlyModel, CLAPModel, HTSATRobertaModel
from .task_config import TaskConfig


def classification_metrics(
    actual: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must have the same length")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Metric labels must be a non-empty unique sequence")
    label_set = set(labels)
    if any(label not in label_set for label in [*actual, *predicted]):
        raise ValueError("Actual and predicted labels must belong to the metric label set")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual, predicted))
        false_positive = sum(a != label and p == label for a, p in zip(actual, predicted))
        false_negative = sum(a == label and p != label for a, p in zip(actual, predicted))
        support = sum(a == label for a in actual)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if support == 0 else true_positive / support
        per_class[label] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
        }
        f1_values.append(f1)
    count = len(actual)
    return {
        "sample_count": count,
        "top1_accuracy": sum(a == p for a, p in zip(actual, predicted)) / count if count else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def evaluate_task(
    model: CLAPModel | HTSATRobertaModel | AudioOnlyModel,
    dataset: TaskAudioDataset,
    task_config: TaskConfig,
    experiment: str,
    device: torch.device,
) -> dict[str, Any]:
    labels = task_config.labels
    if experiment not in task_config.experiments:
        raise ValueError(f"Unknown experiment {experiment!r}")
    definition = task_config.experiment(experiment)
    mode = str(definition["mode"])
    conditioning = str(definition.get("conditioning", "none"))
    if mode not in {"audio_only", "audio_text"}:
        raise ValueError("Evaluation requires a runnable B0-B4 experiment")
    if isinstance(model, AudioOnlyModel) != (mode == "audio_only"):
        raise ValueError("Checkpoint model type does not match the selected experiment")

    tokenizer = (
        PretrainedTokenizer(model.text_model_name, model.config.max_text_tokens)
        if isinstance(model, HTSATRobertaModel)
        else HashTokenizer(model.config.vocabulary_size, model.config.max_text_tokens)
    )
    actual: list[str] = []
    predicted: list[str] = []
    bucket_values: dict[str, tuple[list[str], list[str]]] = {}
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            waveform = sample["waveform"].unsqueeze(0).to(device)
            actual_label = str(sample["label"])
            if isinstance(model, AudioOnlyModel):
                scores = model(waveform).squeeze(0)
            else:
                audio_embedding = model.encode_audio(waveform)
                if conditioning == "none":
                    prompts = [task_config.base_prompt(label) for label in labels]
                    tokens, mask = tokenizer.batch_encode(prompts)
                    text_embeddings = model.encode_text(tokens.to(device), mask.to(device))
                    scores = (audio_embedding @ text_embeddings.T).squeeze(0)
                elif conditioning == "hard_bucket":
                    bucket = sample["condition_bucket"]
                    if not isinstance(bucket, str):
                        raise ValueError("A Leq value is required to evaluate hard-bucket conditioning")
                    prompts = [task_config.conditioned_prompt(label, bucket) for label in labels]
                    tokens, mask = tokenizer.batch_encode(prompts)
                    text_embeddings = model.encode_text(tokens.to(device), mask.to(device))
                    scores = (audio_embedding @ text_embeddings.T).squeeze(0)
                elif conditioning == "soft_bucket":
                    leq_value = float(sample["leq_value"])
                    prompts = [
                        task_config.conditioned_prompt(label, bucket.name)
                        for label in labels
                        for bucket in task_config.buckets
                    ]
                    tokens, mask = tokenizer.batch_encode(prompts)
                    text_embeddings = model.encode_text(tokens.to(device), mask.to(device))
                    text_embeddings = text_embeddings.reshape(
                        len(labels), len(task_config.buckets), -1
                    )
                    weights = torch.tensor(
                        task_config.soft_weights(
                            leq_value,
                            sigma=float(definition.get("soft_sigma_db", task_config.soft_sigma_db)),
                        ),
                        device=device,
                        dtype=text_embeddings.dtype,
                    )
                    weighted_text = functional.normalize(
                        (text_embeddings * weights.view(1, -1, 1)).sum(dim=1), dim=-1
                    )
                    scores = (audio_embedding @ weighted_text.T).squeeze(0)
                else:
                    raise ValueError(f"Unsupported conditioning mode {conditioning!r}")

            prediction = labels[int(scores.argmax().item())]
            actual.append(actual_label)
            predicted.append(prediction)
            bucket = sample["condition_bucket"]
            bucket_name = str(bucket) if bucket is not None else "unconditioned"
            bucket_actual, bucket_predicted = bucket_values.setdefault(bucket_name, ([], []))
            bucket_actual.append(actual_label)
            bucket_predicted.append(prediction)

    metrics = classification_metrics(actual, predicted, labels)
    metrics["by_condition_bucket"] = {
        name: classification_metrics(bucket_actual, bucket_predicted, labels)
        for name, (bucket_actual, bucket_predicted) in sorted(bucket_values.items())
    }
    return metrics


def write_json(payload: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
