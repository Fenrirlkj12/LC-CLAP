from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional
import torchaudio
from torch.utils.data import Dataset

from .config import CLAPConfig
from .task_config import TaskConfig


TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


class HashTokenizer:
    """Stateless tokenizer for the public LC-CLAP configuration."""

    pad_token_id = 0

    def __init__(self, vocabulary_size: int, max_tokens: int) -> None:
        if vocabulary_size < 3:
            raise ValueError("vocabulary_size must be at least 3")
        self.vocabulary_size = vocabulary_size
        self.max_tokens = max_tokens

    def encode(self, text: str) -> torch.Tensor:
        token_ids = []
        for token in TOKEN_PATTERN.findall(text.lower())[: self.max_tokens]:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            token_ids.append(1 + int.from_bytes(digest, "big") % (self.vocabulary_size - 1))
        return torch.tensor(token_ids or [1], dtype=torch.long)

    def batch_encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(text) for text in texts]
        length = min(max(item.numel() for item in encoded), self.max_tokens)
        tokens = torch.full((len(encoded), length), self.pad_token_id, dtype=torch.long)
        for index, item in enumerate(encoded):
            tokens[index, : item.numel()] = item
        return tokens, tokens.ne(self.pad_token_id)


class PretrainedTokenizer:
    """Hugging Face tokenizer adapter for task-specific audio-text models."""

    def __init__(self, model_name: str, max_tokens: int) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Pretrained text models require transformers") from error
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def batch_encode(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].bool()


def load_waveform(audio_path: Path, config: CLAPConfig, training: bool) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")
    waveform = waveform.mean(dim=0)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, config.sample_rate
        )

    target_samples = int(config.sample_rate * config.clip_seconds)
    if waveform.numel() >= target_samples:
        if training and waveform.numel() > target_samples:
            start = torch.randint(
                0, waveform.numel() - target_samples + 1, size=()
            ).item()
        else:
            start = (waveform.numel() - target_samples) // 2
        return waveform[start : start + target_samples]
    return functional.pad(waveform, (0, target_samples - waveform.numel()))


class TaskAudioDataset(Dataset[dict[str, object]]):
    """Audio/label records prepared from the public ten-class task config."""

    def __init__(
        self,
        manifest_path: str | Path,
        audio_config: CLAPConfig,
        task_config: TaskConfig,
        experiment: str,
        training: bool,
        shuffle_scope: str = "within_split",
        shuffle_seed: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audio_config = audio_config
        self.task_config = task_config
        self.requested_experiment = experiment
        self.training = training
        if experiment not in task_config.experiments:
            raise ValueError(f"Unknown experiment {experiment!r}")

        definition = task_config.experiment(experiment)
        self.shuffle_control = None
        if definition.get("mode") == "protocol_only":
            if experiment not in {"B5_shuffle_control", "B7_shuffle_control"}:
                raise ValueError(f"{experiment} is a protocol definition, not a training mode")
            if not training:
                raise ValueError("Shuffle controls may only be applied to training manifests")
            self.shuffle_control = experiment
            control_definition = definition
            self.base_experiment = str(definition["base_experiment"])
            definition = task_config.experiment(self.base_experiment)
            allowed_scopes = control_definition.get("scopes", ("within_split", "global"))
            if shuffle_scope not in allowed_scopes:
                raise ValueError(
                    f"{experiment} supports shuffle scopes: {', '.join(allowed_scopes)}"
                )
            self.shuffle_seed = int(
                shuffle_seed
                if shuffle_seed is not None
                else control_definition.get("shuffle_seed", 3408)
            )
            self._shuffle_scope = shuffle_scope
            if shuffle_scope not in {"within_split", "global"}:
                raise ValueError("shuffle_scope must be within_split or global")
        else:
            self.base_experiment = experiment
            self.shuffle_seed = shuffle_seed
            self._shuffle_scope = shuffle_scope

        self.mode = str(definition["mode"])
        self.conditioning = str(definition.get("conditioning", "none"))
        self.soft_sigma_db = float(
            definition.get("soft_sigma_db", task_config.soft_sigma_db)
        )
        if self.mode not in {"audio_only", "audio_text"}:
            raise ValueError(f"Unsupported training mode {self.mode!r}")
        self.records = self._read_records()

    def _read_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        label_to_index = self.task_config.class_to_index
        needs_leq = self.conditioning in {"hard_bucket", "soft_bucket"} or self.shuffle_control is not None
        with self.manifest_path.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("audio_path"), str):
                    raise ValueError(f"{self.manifest_path}:{line_number} requires a string audio_path")
                label = row.get("label")
                if not isinstance(label, str) or label not in label_to_index:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number} label must match one of the configured classes"
                    )
                value = row.get("leq_value")
                if value is not None:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"{self.manifest_path}:{line_number} leq_value must be numeric")
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(f"{self.manifest_path}:{line_number} leq_value must be finite")
                if needs_leq and value is None:
                    raise ValueError(f"{self.manifest_path}:{line_number} requires leq_value for this experiment")
                audio_path = Path(row["audio_path"])
                if not audio_path.is_absolute():
                    audio_path = self.manifest_path.parent / audio_path
                split = row.get("split")
                if split is not None and not isinstance(split, str):
                    raise ValueError(f"{self.manifest_path}:{line_number} split must be a string when present")
                records.append(
                    {
                        "audio_path": str(audio_path),
                        "label": label,
                        "class_id": label_to_index[label],
                        "leq_value": value,
                        "split": split,
                    }
                )
        if not records:
            raise ValueError(f"No records found in {self.manifest_path}")

        if self.shuffle_control is not None:
            self._shuffle_conditions(records)
        for record in records:
            value = record["leq_value"]
            bucket = record.get("bucket_override")
            if bucket is None and value is not None:
                bucket = self.task_config.bucket_for(float(value))
            record["condition_bucket"] = bucket
            record["condition_id"] = self.task_config.bucket_index(bucket if isinstance(bucket, str) else None)
            record["prompts"] = self._prompts(str(record["label"]), bucket)
            if self.conditioning == "soft_bucket":
                record["soft_weights"] = self.task_config.soft_weights(
                    float(value), sigma=self.soft_sigma_db
                )
            else:
                record["soft_weights"] = [1.0]
        return records

    def _shuffle_groups(self, records: list[dict[str, object]]) -> list[list[int]]:
        groups: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            split = str(record["split"]) if record["split"] is not None else "manifest"
            key = split if self._shuffle_scope == "within_split" else "all"
            groups.setdefault(key, []).append(index)
        return list(groups.values())

    def _shuffle_conditions(self, records: list[dict[str, object]]) -> None:
        rng = random.Random(self.shuffle_seed)
        groups = self._shuffle_groups(records)
        if self.shuffle_control == "B5_shuffle_control":
            definition = self.task_config.experiment(self.shuffle_control)
            ratio = float(definition.get("shuffle_ratio", 1.0))
            if not 0 <= ratio <= 1:
                raise ValueError("B5 shuffle_ratio must be between zero and one")
            for indices in groups:
                count = min(len(indices), round(len(indices) * ratio))
                if count < 2:
                    continue
                selected = rng.sample(indices, count)
                values = [records[index]["leq_value"] for index in selected]
                rng.shuffle(values)
                for index, value in zip(selected, values):
                    records[index]["leq_value"] = value
        else:
            for indices in groups:
                buckets = [self.task_config.bucket_for(float(records[index]["leq_value"])) for index in indices]
                rng.shuffle(buckets)
                for index, bucket in zip(indices, buckets):
                    records[index]["bucket_override"] = bucket

    def _prompts(self, label: str, bucket: object) -> list[str]:
        if self.mode == "audio_only":
            return []
        if self.conditioning == "none":
            return [self.task_config.base_prompt(label)]
        if self.conditioning == "hard_bucket":
            if not isinstance(bucket, str):
                raise ValueError("A Leq bucket is required for hard-bucket conditioning")
            return [self.task_config.conditioned_prompt(label, bucket)]
        if self.conditioning == "soft_bucket":
            return [
                self.task_config.conditioned_prompt(label, item.name)
                for item in self.task_config.buckets
            ]
        raise ValueError(f"Unsupported conditioning mode {self.conditioning!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        value = record["leq_value"]
        return {
            "waveform": load_waveform(Path(record["audio_path"]), self.audio_config, self.training),
            "label": record["label"],
            "class_id": record["class_id"],
            "leq_value": -1.0 if value is None else float(value),
            "condition_bucket": record["condition_bucket"],
            "condition_id": record["condition_id"],
            "prompts": record["prompts"],
            "soft_weights": record["soft_weights"],
        }


def collate_task_batch(
    batch: list[dict[str, object]], tokenizer: HashTokenizer | PretrainedTokenizer | None
) -> dict[str, object]:
    waveforms = torch.stack([item["waveform"] for item in batch])
    prompt_rows = [item["prompts"] for item in batch]
    prompt_count = len(prompt_rows[0])
    if any(len(row) != prompt_count for row in prompt_rows):
        raise ValueError("Each item in a batch must have the same number of prompts")
    if prompt_count:
        if tokenizer is None:
            raise ValueError("A tokenizer is required for audio-text batches")
        prompts = [str(prompt) for row in prompt_rows for prompt in row]
        tokens, attention_mask = tokenizer.batch_encode(prompts)
    else:
        tokens = torch.empty((0, 0), dtype=torch.long)
        attention_mask = torch.empty((0, 0), dtype=torch.bool)
    return {
        "waveforms": waveforms,
        "tokens": tokens,
        "attention_mask": attention_mask,
        "prompt_count": prompt_count,
        "soft_weights": torch.tensor([item["soft_weights"] for item in batch], dtype=torch.float32),
        "class_ids": torch.tensor([item["class_id"] for item in batch], dtype=torch.long),
        "condition_ids": torch.tensor([item["condition_id"] for item in batch], dtype=torch.long),
        "labels": [str(item["label"]) for item in batch],
        "condition_buckets": [item["condition_bucket"] for item in batch],
        "leq_values": torch.tensor([item["leq_value"] for item in batch], dtype=torch.float32),
    }
