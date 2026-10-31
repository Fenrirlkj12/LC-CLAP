#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import HashTokenizer, PretrainedTokenizer, load_waveform
from src.model import AudioOnlyModel, CLAPModel, HTSATRobertaModel, load_model
from src.task_config import TaskConfig, load_task_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank text labels for one audio file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--task-config", default=None, help="Use configured ten-class labels and prompts")
    parser.add_argument("--experiment", default=None, help="Override the checkpoint experiment name")
    parser.add_argument("--leq", type=float, default=None, help="Leq in dB for B3/B4 conditioning")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    waveform = load_waveform(args.audio, model.config, training=False).unsqueeze(0).to(device)
    with torch.inference_mode():
        task_config = load_task_config(args.task_config) if args.task_config else None
        metadata = getattr(model, "checkpoint_metadata", {})
        if task_config is not None:
            experiment = args.experiment or metadata.get("base_experiment") or metadata.get("experiment")
            if not isinstance(experiment, str):
                raise ValueError("--experiment is required when checkpoint metadata has no task experiment")
            reported_experiment = args.experiment or metadata.get("experiment") or experiment
            definition = task_config.experiment(experiment)
            if definition.get("mode") == "protocol_only":
                experiment = str(definition["base_experiment"])
                definition = task_config.experiment(experiment)
            labels = args.labels or task_config.labels
            unknown = set(labels) - set(task_config.labels)
            if unknown:
                raise ValueError(f"Labels are not in the task config: {sorted(unknown)}")
            if isinstance(model, AudioOnlyModel) != (definition.get("mode") == "audio_only"):
                raise ValueError("Checkpoint model type does not match the selected experiment")
            friendly_names = task_config.friendly_names
            if isinstance(model, AudioOnlyModel):
                all_scores = model(waveform).squeeze(0)
                indices = [task_config.class_to_index[label] for label in labels]
                scores = all_scores[indices]
            else:
                scores = _task_text_scores(
                    model, task_config, labels, definition, args.leq, waveform, device
                )
            model_experiment = reported_experiment
        elif isinstance(model, AudioOnlyModel):
            all_labels = metadata.get("labels")
            if not isinstance(all_labels, list) or not all(isinstance(x, str) for x in all_labels):
                raise ValueError("Audio-only checkpoint requires a task config or stored labels")
            labels = args.labels or all_labels
            if any(label not in all_labels for label in labels):
                raise ValueError("Requested label is not present in checkpoint metadata")
            scores = model(waveform).squeeze(0)[[all_labels.index(label) for label in labels]]
            stored_names = metadata.get("friendly_names", {})
            friendly_names = {
                label: stored_names.get(label, label) for label in labels
            }
            model_experiment = metadata.get("experiment")
        else:
            if not args.labels:
                raise ValueError("--labels is required for generic audio-text checkpoints")
            labels = args.labels
            tokenizer = (
                PretrainedTokenizer(model.text_model_name, model.config.max_text_tokens)
                if isinstance(model, HTSATRobertaModel)
                else HashTokenizer(model.config.vocabulary_size, model.config.max_text_tokens)
            )
            tokens, attention_mask = tokenizer.batch_encode(labels)
            audio_embedding = model.encode_audio(waveform)
            text_embeddings = model.encode_text(tokens.to(device), attention_mask.to(device))
            scores = (audio_embedding @ text_embeddings.T).squeeze(0)
            friendly_names = {label: label for label in labels}
            model_experiment = metadata.get("experiment")

        top_count = min(args.top_k, len(labels))
        ordering = scores.argsort(descending=True)[:top_count].cpu().tolist()
        results = [
            {
                "label": labels[index],
                "name": friendly_names[labels[index]],
                "score": round(float(scores[index]), 6),
            }
            for index in ordering
        ]
    print(
        json.dumps(
            {"audio": str(args.audio), "experiment": model_experiment, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )


def _task_text_scores(
    model: CLAPModel | HTSATRobertaModel,
    task_config: TaskConfig,
    labels: list[str],
    definition: dict[str, object],
    leq_value: float | None,
    waveform: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    conditioning = str(definition.get("conditioning", "none"))
    tokenizer = (
        PretrainedTokenizer(model.text_model_name, model.config.max_text_tokens)
        if isinstance(model, HTSATRobertaModel)
        else HashTokenizer(model.config.vocabulary_size, model.config.max_text_tokens)
    )
    audio_embedding = model.encode_audio(waveform)
    if conditioning == "none":
        prompts = [task_config.base_prompt(label) for label in labels]
        tokens, mask = tokenizer.batch_encode(prompts)
        text_embeddings = model.encode_text(tokens.to(device), mask.to(device))
        return (audio_embedding @ text_embeddings.T).squeeze(0)
    if leq_value is None:
        raise ValueError("--leq in dB is required for B3/B4 retrieval")
    if conditioning == "hard_bucket":
        bucket = task_config.bucket_for(leq_value)
        prompts = [task_config.conditioned_prompt(label, str(bucket)) for label in labels]
        tokens, mask = tokenizer.batch_encode(prompts)
        text_embeddings = model.encode_text(tokens.to(device), mask.to(device))
        return (audio_embedding @ text_embeddings.T).squeeze(0)
    if conditioning == "soft_bucket":
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
        weighted_text = torch.nn.functional.normalize(
            (text_embeddings * weights.view(1, -1, 1)).sum(dim=1), dim=-1
        )
        return (audio_embedding @ weighted_text.T).squeeze(0)
    raise ValueError(f"Unsupported conditioning mode {conditioning!r}")


if __name__ == "__main__":
    main()
