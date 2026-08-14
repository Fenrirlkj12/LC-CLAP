from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as functional
import torchaudio
from torch.utils.data import Dataset

from .config import CLAPConfig
from .lc_pattern import LCMethodProfile, ids_for


TOKEN_PATTERN = re.compile(r"[\w']+", flags=re.UNICODE)


class HashTokenizer:
    """Stateless tokenizer suitable for a small reference implementation."""

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


class AudioTextDataset(Dataset[dict[str, object]]):
    """Reads paired records with optional public LC-CLAP pattern metadata."""

    def __init__(
        self,
        manifest_path: str | Path,
        config: CLAPConfig,
        training: bool,
        method_profile: LCMethodProfile | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.training = training
        self.method_profile = method_profile
        self.records = self._read_records()

    def _read_records(self) -> list[dict[str, object]]:
        records = []
        with self.manifest_path.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get("audio_path"), str) or not isinstance(record.get("text"), str):
                    raise ValueError(
                        f"{self.manifest_path}:{line_number} requires string audio_path and text fields"
                    )
                label = record.get("label")
                if label is not None and not isinstance(label, str):
                    raise ValueError(f"{self.manifest_path}:{line_number} label must be a string")
                leq_value = record.get("leq_value")
                if leq_value is not None and not isinstance(leq_value, (int, float)):
                    raise ValueError(f"{self.manifest_path}:{line_number} leq_value must be numeric")
                audio_path = Path(record["audio_path"])
                if not audio_path.is_absolute():
                    audio_path = self.manifest_path.parent / audio_path
                text, bucket = (record["text"], None)
                if self.method_profile is not None:
                    text, bucket = self.method_profile.build_training_text(
                        label, None if leq_value is None else float(leq_value), record["text"]
                    )
                records.append(
                    {
                        "audio_path": str(audio_path),
                        "text": text,
                        "label": label,
                        "condition_bucket": bucket,
                    }
                )
        if not records:
            raise ValueError(f"No records found in {self.manifest_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            "waveform": load_waveform(Path(record["audio_path"]), self.config, self.training),
            "text": record["text"],
            "label": record["label"],
            "condition_bucket": record["condition_bucket"],
        }


def collate_batch(
    batch: list[dict[str, object]], tokenizer: HashTokenizer
) -> dict[str, object]:
    waveforms = torch.stack([item["waveform"] for item in batch])
    tokens, attention_mask = tokenizer.batch_encode([str(item["text"]) for item in batch])
    labels = [item["label"] if isinstance(item["label"], str) else None for item in batch]
    condition_buckets = [
        item["condition_bucket"] if isinstance(item["condition_bucket"], str) else None
        for item in batch
    ]
    return {
        "waveforms": waveforms,
        "tokens": tokens,
        "attention_mask": attention_mask,
        "class_ids": ids_for(labels),
        "condition_ids": ids_for(condition_buckets),
        "labels": labels,
        "condition_buckets": condition_buckets,
    }
