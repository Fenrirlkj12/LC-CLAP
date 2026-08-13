from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchaudio
from torch import nn

from .config import CLAPConfig


class AudioEncoder(nn.Module):
    """Mel-spectrogram front end followed by a compact convolutional encoder."""

    def __init__(self, config: CLAPConfig) -> None:
        super().__init__()
        width = config.encoder_width
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )
        self.features = nn.Sequential(
            nn.Conv2d(1, width // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width // 2),
            nn.GELU(),
            nn.Conv2d(width // 2, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output = nn.Linear(width, config.embedding_dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveforms).clamp_min(1e-5).log()
        features = self.features(mel.unsqueeze(1)).flatten(start_dim=1)
        return self.output(features)


class TextEncoder(nn.Module):
    """Token embedding and Transformer encoder used for paired descriptions."""

    def __init__(self, config: CLAPConfig) -> None:
        super().__init__()
        if config.encoder_width % config.text_heads != 0:
            raise ValueError("encoder_width must be divisible by text_heads")
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.encoder_width, padding_idx=0
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.max_text_tokens, config.encoder_width)
        )
        nn.init.normal_(self.position_embedding, std=0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_width,
            nhead=config.text_heads,
            dim_feedforward=config.encoder_width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.text_layers)
        self.output = nn.Linear(config.encoder_width, config.embedding_dim)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("Token sequence exceeds max_text_tokens")
        encoded = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.transformer(
            encoded, src_key_padding_mask=~attention_mask.bool()
        )
        weights = attention_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output(pooled)


class CLAPModel(nn.Module):
    """Dual encoder with normalized embeddings and a learned logit scale."""

    def __init__(self, config: CLAPConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.audio_encoder(waveforms), dim=-1)

    def encode_text(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return functional.normalize(self.text_encoder(tokens, attention_mask), dim=-1)

    def forward(
        self, waveforms: torch.Tensor, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        audio_embeddings = self.encode_audio(waveforms)
        text_embeddings = self.encode_text(tokens, attention_mask)
        return self.logit_scale.exp().clamp(max=100) * audio_embeddings @ text_embeddings.T


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE objective over aligned audio-text pairs in one batch."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Contrastive logits must be a square audio-text matrix")
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (functional.cross_entropy(logits, targets) + functional.cross_entropy(logits.T, targets)) / 2


def save_checkpoint(
    destination: str | Path,
    model: CLAPModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        destination,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: CLAPModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[int, int]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"]), int(payload["global_step"])


def load_model(checkpoint_path: str | Path, device: torch.device | str) -> CLAPModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CLAPModel(CLAPConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
