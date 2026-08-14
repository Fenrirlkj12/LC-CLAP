"""Configurable LC-CLAP condition modeling and evaluation components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class ConditionBucket:
    """One ordered bucket in a scalar acoustic-condition representation."""

    name: str
    upper_bound: float | None


@dataclass(frozen=True)
class PromptPolicy:
    """Builds text descriptions from a class label and condition bucket."""

    templates: tuple[str, ...] = (
        "a recording of {label} under {condition} conditions",
    )

    def build(self, label: str, condition: str) -> list[str]:
        return [template.format(label=label, condition=condition) for template in self.templates]


@dataclass(frozen=True)
class HardNegativePolicy:
    """Emphasizes mismatched classes that share the same known condition."""

    enabled: bool = True
    scale: float = 1.2

    def apply(
        self,
        logits: torch.Tensor,
        class_ids: torch.Tensor | None,
        condition_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enabled or class_ids is None or condition_ids is None:
            return logits
        if class_ids.shape[0] != logits.shape[0] or condition_ids.shape[0] != logits.shape[0]:
            raise ValueError("Hard-negative metadata must match the contrastive batch size")
        if self.scale < 1.0:
            raise ValueError("Hard-negative scale must be at least 1")

        same_condition = condition_ids[:, None].eq(condition_ids[None, :])
        known_condition = condition_ids[:, None].ge(0) & condition_ids[None, :].ge(0)
        different_class = class_ids[:, None].ne(class_ids[None, :])
        diagonal = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask = same_condition & known_condition & different_class & ~diagonal
        return torch.where(mask, logits + math.log(self.scale), logits)


@dataclass(frozen=True)
class EvaluationProtocol:
    """Five-run evaluation protocol with distinct random seeds."""

    seed_ids: tuple[int, ...] = (101, 203, 307, 401, 509)

    def validate(self) -> None:
        if len(self.seed_ids) != 5 or len(set(self.seed_ids)) != 5:
            raise ValueError("The evaluation protocol requires five distinct seed identifiers")


@dataclass(frozen=True)
class LCMethodProfile:
    """LC-CLAP configuration for conditioning, hard negatives, and evaluation."""

    buckets: tuple[ConditionBucket, ...] = (
        ConditionBucket("low_context", 0.35),
        ConditionBucket("medium_context", 0.7),
        ConditionBucket("high_context", None),
    )
    prompts: PromptPolicy = field(default_factory=PromptPolicy)
    hard_negatives: HardNegativePolicy = field(default_factory=HardNegativePolicy)
    evaluation: EvaluationProtocol = field(default_factory=EvaluationProtocol)

    def __post_init__(self) -> None:
        if not self.buckets or self.buckets[-1].upper_bound is not None:
            raise ValueError("The final bucket must have upper_bound=None")
        finite_bounds = [bucket.upper_bound for bucket in self.buckets[:-1]]
        if any(bound is None for bound in finite_bounds):
            raise ValueError("Only the final bucket may be open-ended")
        if finite_bounds != sorted(finite_bounds):
            raise ValueError("Bucket boundaries must be ordered")
        self.evaluation.validate()

    def bucket_for(self, condition_value: float | None) -> str | None:
        if condition_value is None:
            return None
        for bucket in self.buckets:
            if bucket.upper_bound is None or condition_value <= bucket.upper_bound:
                return bucket.name
        raise RuntimeError("An open-ended final bucket is required")

    def build_training_text(
        self, label: str | None, condition_value: float | None, fallback: str
    ) -> tuple[str, str | None]:
        condition = self.bucket_for(condition_value)
        if label is None or condition is None:
            return fallback, condition
        return self.prompts.build(label, condition)[0], condition

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_method_profile(profile_path: str | Path) -> LCMethodProfile:
    """Load an LC-CLAP configuration from a JSON document."""
    with Path(profile_path).open(encoding="utf-8") as profile_file:
        source = json.load(profile_file)
    buckets = tuple(
        ConditionBucket(name=item["name"], upper_bound=item.get("upper_bound"))
        for item in source["buckets"]
    )
    prompts = PromptPolicy(templates=tuple(source["prompts"]["templates"]))
    hard_negatives = HardNegativePolicy(**source["hard_negatives"])
    evaluation = EvaluationProtocol(seed_ids=tuple(source["evaluation"]["seed_ids"]))
    return LCMethodProfile(
        buckets=buckets,
        prompts=prompts,
        hard_negatives=hard_negatives,
        evaluation=evaluation,
    )


def ids_for(values: Iterable[str | None]) -> torch.Tensor:
    """Map batch-local names to ids; missing values remain -1 and are ignored."""
    mapping: dict[str, int] = {}
    output = []
    for value in values:
        if value is None:
            output.append(-1)
            continue
        if value not in mapping:
            mapping[value] = len(mapping)
        output.append(mapping[value])
    return torch.tensor(output, dtype=torch.long)
