from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from .config import CLAPConfig


def _init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


def _init_batch_norm(layer: nn.BatchNorm2d) -> None:
    nn.init.zeros_(layer.bias)
    nn.init.ones_(layer.weight)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_batch_norm(self.bn1)
        _init_batch_norm(self.bn2)

    def forward(self, inputs: torch.Tensor, pool_size: tuple[int, int]) -> torch.Tensor:
        outputs = functional.relu_(self.bn1(self.conv1(inputs)))
        outputs = functional.relu_(self.bn2(self.conv2(outputs)))
        return functional.avg_pool2d(outputs, kernel_size=pool_size)


class Cnn14Backbone(nn.Module):
    """PANNs CNN14 audio encoder (AudioSet head retained for checkpoint compatibility)."""

    output_dim = 2048

    def __init__(self, config: CLAPConfig) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=50,
            fmax=min(14_000, config.sample_rate / 2),
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(config.n_mels)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)
        _init_batch_norm(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    def load_panns_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        if not isinstance(state, dict):
            raise ValueError("CNN14 checkpoint must contain a model state dictionary")
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        required = {"conv_block1.conv1.weight", "conv_block6.conv2.weight", "fc1.weight"}
        loaded = set(self.state_dict()) - set(missing)
        if not required.issubset(loaded):
            raise ValueError("Checkpoint does not match the PANNs CNN14 architecture")
        if unexpected:
            print(f"CNN14 checkpoint ignored {len(unexpected)} unexpected parameter(s)")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.spectrogram_extractor(waveforms)
        outputs = self.logmel_extractor(outputs)
        outputs = self.bn0(outputs.transpose(1, 3)).transpose(1, 3)
        if self.training:
            outputs = self.spec_augmenter(outputs)

        outputs = self.conv_block1(outputs, (2, 2))
        outputs = functional.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block2(outputs, (2, 2))
        outputs = functional.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block3(outputs, (2, 2))
        outputs = functional.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block4(outputs, (2, 2))
        outputs = functional.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block5(outputs, (2, 2))
        outputs = functional.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block6(outputs, (1, 1))
        outputs = functional.dropout(outputs, p=0.2, training=self.training)
        outputs = outputs.mean(dim=3)
        outputs = outputs.amax(dim=2) + outputs.mean(dim=2)
        outputs = functional.dropout(outputs, p=0.5, training=self.training)
        outputs = functional.relu_(self.fc1(outputs))
        return functional.dropout(outputs, p=0.5, training=self.training)


class HtsatTinyBackbone(nn.Module):
    """HTSAT-tiny audio encoder using the Hugging Face Transformers implementation."""

    output_dim = 768

    def __init__(self, config: CLAPConfig, pretrained_audio: str | None = None) -> None:
        super().__init__()
        try:
            from transformers import ClapAudioConfig, ClapAudioModel, ClapFeatureExtractor
        except ImportError as error:
            raise RuntimeError("HTSAT-tiny requires the transformers package") from error

        self.sample_rate = config.sample_rate
        if pretrained_audio:
            self.encoder = ClapAudioModel.from_pretrained(pretrained_audio)
            self.feature_extractor = ClapFeatureExtractor.from_pretrained(pretrained_audio)
            self.output_dim = int(self.encoder.config.hidden_size)
        else:
            audio_config = ClapAudioConfig(
                num_mel_bins=config.n_mels,
                spec_size=256,
                patch_size=4,
                patch_stride=[4, 4],
                num_classes=527,
                hidden_size=768,
                depths=[2, 2, 6, 2],
                num_attention_heads=[4, 8, 16, 32],
                patch_embeds_hidden_size=96,
                drop_path_rate=0.1,
                enable_fusion=False,
            )
            self.encoder = ClapAudioModel(audio_config)
            self.feature_extractor = ClapFeatureExtractor(
                feature_size=config.n_mels,
                sampling_rate=config.sample_rate,
                hop_length=config.hop_length,
                max_length_s=config.clip_seconds,
                fft_window_size=config.n_fft,
                frequency_min=50,
                frequency_max=min(14_000, config.sample_rate / 2),
                truncation="rand_trunc",
                padding="repeatpad",
            )
            self.output_dim = int(audio_config.hidden_size)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        audio = [waveform.detach().cpu().numpy() for waveform in waveforms]
        features = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        encoded = self.encoder(input_features=features.input_features.to(waveforms.device))
        return encoded.pooler_output


def build_audio_backbone(
    backbone: str,
    config: CLAPConfig,
    pretrained_audio: str | None = None,
) -> nn.Module:
    if backbone == "cnn14":
        model = Cnn14Backbone(config)
        if pretrained_audio:
            model.load_panns_checkpoint(pretrained_audio)
        return model
    if backbone == "htsat_tiny":
        return HtsatTinyBackbone(config, pretrained_audio=pretrained_audio)
    raise ValueError(f"Unsupported audio-only backbone: {backbone}")
