from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CLAPConfig:
    sample_rate: int = 48_000
    clip_seconds: float = 10.0
    n_mels: int = 64
    n_fft: int = 1_024
    hop_length: int = 480
    vocabulary_size: int = 32_768
    max_text_tokens: int = 64
    encoder_width: int = 256
    embedding_dim: int = 512
    text_layers: int = 4
    text_heads: int = 8
    dropout: float = 0.1
    initial_temperature: float = 0.07
