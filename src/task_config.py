from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskClass:
    id: int
    name: str
    friendly_name: str


@dataclass(frozen=True)
class LeqBucket:
    name: str
    max_inclusive: float | None
    center_db: float
    prompt_phrase: str


@dataclass(frozen=True)
class TaskConfig:
    name: str
    sample_rate_hz: int
    clip_seconds: float
    classes: tuple[TaskClass, ...]
    buckets: tuple[LeqBucket, ...]
    soft_sigma_db: float
    class_only_template: str
    conditioned_template: str
    prompt_matrix: dict[str, dict[str, str]]
    experiments: dict[str, dict[str, Any]]
    training_defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return [item.name for item in self.classes]

    @property
    def class_to_index(self) -> dict[str, int]:
        return {item.name: item.id for item in self.classes}

    @property
    def friendly_names(self) -> dict[str, str]:
        return {item.name: item.friendly_name for item in self.classes}

    def bucket_for(self, value: float | None) -> str | None:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Leq value must be finite")
        for bucket in self.buckets:
            if bucket.max_inclusive is None or numeric <= bucket.max_inclusive:
                return bucket.name
        raise RuntimeError("The final Leq bucket must be open-ended")

    def base_prompt(self, label: str) -> str:
        return self.class_only_template.format(label=self.friendly_names[label])

    def conditioned_prompt(self, label: str, bucket: str) -> str:
        return self.prompt_matrix[label][bucket]

    def soft_weights(self, value: float, sigma: float | None = None) -> list[float]:
        width = self.soft_sigma_db if sigma is None else float(sigma)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Soft Leq sigma must be positive")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Leq value must be finite")
        weights = [
            math.exp(-((numeric - bucket.center_db) ** 2) / (2.0 * width**2))
            for bucket in self.buckets
        ]
        total = sum(weights)
        if total <= 1e-12:
            return [1.0 / len(weights)] * len(weights)
        return [weight / total for weight in weights]

    def bucket_index(self, name: str | None) -> int:
        if name is None:
            return -1
        for index, bucket in enumerate(self.buckets):
            if bucket.name == name:
                return index
        raise ValueError(f"Unknown Leq bucket {name!r}")

    def experiment(self, name: str) -> dict[str, Any]:
        try:
            return self.experiments[name]
        except KeyError as error:
            raise ValueError(f"Unknown experiment {name!r}; choose from {', '.join(self.experiments)}") from error


def load_task_config(path: str | Path) -> TaskConfig:
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = tuple(
        TaskClass(
            id=int(item["id"]),
            name=str(item["name"]),
            friendly_name=str(item["friendly_name"]),
        )
        for item in source["classes"]
    )
    buckets = tuple(
        LeqBucket(
            name=str(item["name"]),
            max_inclusive=None if item.get("max_inclusive") is None else float(item["max_inclusive"]),
            center_db=float(item["center_db"]),
            prompt_phrase=str(item["prompt_phrase"]),
        )
        for item in source["leq"]["buckets"]
    )
    prompts = source["prompts"]
    matrix = prompts["matrix"]
    experiment_definitions = source["experiments"]
    config = TaskConfig(
        name=str(source["name"]),
        sample_rate_hz=int(source["sample_rate_hz"]),
        clip_seconds=float(source["clip_seconds"]),
        classes=classes,
        buckets=buckets,
        soft_sigma_db=float(source["leq"]["soft_sigma_db"]),
        class_only_template=str(prompts["class_only_template"]),
        conditioned_template=str(prompts["conditioned_template"]),
        prompt_matrix=matrix,
        experiments=experiment_definitions,
        training_defaults=dict(source.get("training_defaults", {})),
    )
    _validate_task_config(config)
    return config


def _validate_task_config(config: TaskConfig) -> None:
    if config.sample_rate_hz <= 0 or config.clip_seconds <= 0:
        raise ValueError("sample rate and clip duration must be positive")
    if not config.classes or [item.id for item in config.classes] != list(range(len(config.classes))):
        raise ValueError("Class ids must be contiguous and ordered from zero")
    if len(set(config.labels)) != len(config.labels):
        raise ValueError("Class names must be unique")
    if not math.isfinite(config.soft_sigma_db) or config.soft_sigma_db <= 0:
        raise ValueError("Soft Leq sigma must be a positive finite value")
    if not config.buckets or config.buckets[-1].max_inclusive is not None:
        raise ValueError("The final Leq bucket must be open-ended")
    finite_bounds = [item.max_inclusive for item in config.buckets[:-1]]
    if any(value is None or not math.isfinite(value) for value in finite_bounds):
        raise ValueError("Leq bucket boundaries must be ordered")
    if any(left >= right for left, right in zip(finite_bounds, finite_bounds[1:])):
        raise ValueError("Leq bucket boundaries must be strictly increasing")
    if len({item.name for item in config.buckets}) != len(config.buckets):
        raise ValueError("Leq bucket names must be unique")
    if any(not math.isfinite(item.center_db) for item in config.buckets):
        raise ValueError("Leq bucket centers must be finite")
    required_defaults = {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "workers",
        "save_every",
        "seed",
        "embedding_dim",
    }
    if not required_defaults.issubset(config.training_defaults):
        missing = sorted(required_defaults - set(config.training_defaults))
        raise ValueError(f"training_defaults missing keys: {', '.join(missing)}")
    bucket_names = [item.name for item in config.buckets]
    if set(config.prompt_matrix) != set(config.labels):
        raise ValueError("Prompt matrix must contain exactly the configured classes")
    for item in config.classes:
        row = config.prompt_matrix[item.name]
        if set(row) != set(bucket_names):
            raise ValueError(f"Prompt matrix row for {item.name!r} must contain every Leq bucket")
        for bucket in config.buckets:
            expected = config.conditioned_template.format(
                label=item.friendly_name,
                condition=bucket.prompt_phrase,
            )
            if row[bucket.name] != expected:
                raise ValueError(f"Prompt matrix entry differs from its template: {item.name}/{bucket.name}")
    for name in ("B0", "B1", "B2", "B3", "B4"):
        if name not in config.experiments:
            raise ValueError(f"Missing required experiment configuration {name}")
