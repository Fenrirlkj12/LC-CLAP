from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchaudio
from torch import nn

from .audio_backbones import build_audio_backbone
from .config import CLAPConfig


@dataclass(frozen=True)
class HardNegativePolicy:
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

    model_type = "contrastive"

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


class HTSATRobertaModel(nn.Module):
    """HTSAT audio and RoBERTa text dual encoder for B2-B4."""

    model_type = "htsat_roberta_contrastive"

    def __init__(
        self,
        config: CLAPConfig,
        audio_model_name: str = "laion/clap-htsat-unfused",
        text_model_name: str = "roberta-base",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("HTSAT/RoBERTa models require transformers") from error

        self.config = config
        self.audio_model_name = audio_model_name
        self.text_model_name = text_model_name
        self.audio_encoder = build_audio_backbone(
            "htsat_tiny", config, pretrained_audio=audio_model_name
        )
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = int(self.text_encoder.config.hidden_size)
        audio_dim = int(self.audio_encoder.output_dim)
        self.audio_projection = nn.Linear(audio_dim, config.embedding_dim)
        self.text_projection = nn.Linear(text_dim, config.embedding_dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.initial_temperature))
        )

    def encode_audio(self, waveforms: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.audio_projection(self.audio_encoder(waveforms)), dim=-1)

    def encode_text(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=tokens, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return functional.normalize(self.text_projection(pooled), dim=-1)

    def forward(
        self, waveforms: torch.Tensor, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        audio_embeddings = self.encode_audio(waveforms)
        text_embeddings = self.encode_text(tokens, attention_mask)
        return self.logit_scale.exp().clamp(max=100) * audio_embeddings @ text_embeddings.T


class AudioOnlyModel(nn.Module):
    """Supervised classifier with a configured audio-only backbone."""

    model_type = "audio_only"

    def __init__(
        self,
        config: CLAPConfig,
        num_classes: int,
        backbone: str = "compact",
        pretrained_audio: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        if backbone == "compact":
            self.audio_encoder = AudioEncoder(config)
            output_dim = config.embedding_dim
        else:
            self.audio_encoder = build_audio_backbone(
                backbone, config, pretrained_audio=pretrained_audio
            )
            output_dim = int(self.audio_encoder.output_dim)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.audio_encoder(waveforms))


def contrastive_loss(
    logits: torch.Tensor,
    class_ids: torch.Tensor | None = None,
    condition_ids: torch.Tensor | None = None,
    hard_negative_policy: HardNegativePolicy | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE objective over aligned audio-text pairs in one batch."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Contrastive logits must be a square audio-text matrix")
    if hard_negative_policy is not None:
        logits = hard_negative_policy.apply(logits, class_ids, condition_ids)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (functional.cross_entropy(logits, targets) + functional.cross_entropy(logits.T, targets)) / 2


def save_checkpoint(
    destination: str | Path,
    model: CLAPModel | HTSATRobertaModel | AudioOnlyModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metadata: dict[str, object] | None = None,
) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": model.model_type,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metadata": metadata or {},
    }
    if isinstance(model, AudioOnlyModel):
        payload["num_classes"] = model.classifier.out_features
        payload["audio_backbone"] = model.backbone
    if isinstance(model, HTSATRobertaModel):
        payload["audio_model_name"] = model.audio_model_name
        payload["text_model_name"] = model.text_model_name
    torch.save(payload, destination)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: CLAPModel | HTSATRobertaModel | AudioOnlyModel,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[int, int]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["epoch"]), int(payload["global_step"])


def load_model(
    checkpoint_path: str | Path, device: torch.device | str
) -> CLAPModel | HTSATRobertaModel | AudioOnlyModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = CLAPConfig(**payload["model_config"])
    if payload.get("model_type") == AudioOnlyModel.model_type:
        state = payload["model_state"]
        class_count = payload.get("num_classes")
        if class_count is None:
            class_count = state["classifier.weight"].shape[0]
        model = AudioOnlyModel(
            config,
            int(class_count),
            backbone=str(payload.get("audio_backbone", "compact")),
        ).to(device)
    elif payload.get("model_type") == HTSATRobertaModel.model_type:
        model = HTSATRobertaModel(
            config,
            audio_model_name=str(payload.get("audio_model_name", "laion/clap-htsat-unfused")),
            text_model_name=str(payload.get("text_model_name", "roberta-base")),
        ).to(device)
    else:
        model = CLAPModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    model.checkpoint_metadata = payload.get("metadata", {})
    return model
