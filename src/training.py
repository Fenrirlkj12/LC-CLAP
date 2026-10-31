from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .config import CLAPConfig
from .data import PretrainedTokenizer, TaskAudioDataset, collate_task_batch
from .model import (
    AudioOnlyModel,
    HTSATRobertaModel,
    contrastive_loss,
    load_checkpoint,
    save_checkpoint,
)
from .task_config import TaskConfig


def train_task(
    manifest_path: str | Path,
    output_directory: str | Path,
    config: CLAPConfig,
    task_config: TaskConfig,
    experiment: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    device: torch.device,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    save_every: int = 1,
    shuffle_scope: str = "within_split",
    shuffle_seed: int | None = None,
    pretrained_audio: str | None = None,
    pretrained_text: str | None = None,
) -> None:
    if epochs < 1 or batch_size < 2 or learning_rate <= 0 or weight_decay < 0 or workers < 0 or save_every < 1:
        raise ValueError("training hyperparameters are invalid")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = TaskAudioDataset(
        manifest_path,
        config,
        task_config,
        experiment,
        training=True,
        shuffle_scope=shuffle_scope,
        shuffle_seed=shuffle_seed,
    )
    if dataset.mode == "audio_text" and len(dataset) < 2:
        raise ValueError("Audio-text contrastive training requires at least two manifest records")
    if dataset.mode == "audio_text":
        if not pretrained_audio or not pretrained_text:
            raise ValueError(
                "B2-B4 require explicit --pretrained-audio and --pretrained-text model IDs"
            )
        tokenizer = PretrainedTokenizer(pretrained_text, config.max_text_tokens)
    else:
        tokenizer = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_task_batch(batch, tokenizer),
    )

    model: HTSATRobertaModel | AudioOnlyModel
    if dataset.mode == "audio_only":
        definition = task_config.experiment(dataset.base_experiment)
        backbone = str(definition.get("audio_only_backbone", "compact"))
        model = AudioOnlyModel(
            config,
            num_classes=len(task_config.classe),
            backbone=backbone,
            pretrained_audio=pretrained_audio if resume_checkpoint is None else None,
        ).to(device)
    else:
        model = HTSATRobertaModel(
            config,
            audio_model_name=pretrained_audio,
            text_model_name=pretrained_text,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch, global_step = 0, 0
    if resume_checkpoint is not None:
        completed_epoch, global_step = load_checkpoint(
            resume_checkpoint, model, optimizer, device
        )
        start_epoch = completed_epoch + 1
    if start_epoch >= epochs:
        raise ValueError("epochs must be greater than the last completed checkpoint epoch")

    definition = task_config.experiment(dataset.base_experiment)
    metadata = {
        "task_config": task_config.name,
        "experiment": dataset.requested_experiment,
        "base_experiment": dataset.base_experiment,
        "labels": task_config.labels,
        "friendly_names": task_config.friendly_names,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "shuffle_scope": shuffle_scope if dataset.shuffle_control else None,
        "shuffle_seed": dataset.shuffle_seed if dataset.shuffle_control else None,
        "soft_sigma_db": dataset.soft_sigma_db if dataset.conditioning == "soft_bucket" else None,
        "audio_backbone": model.backbone if isinstance(model, AudioOnlyModel) else None,
        "audio_model_name": model.audio_model_name if isinstance(model, HTSATRobertaModel) else None,
        "text_model_name": model.text_model_name if isinstance(model, HTSATRobertaModel) else None,
    }
    hard_negative = HardNegativePolicy(
        enabled=bool(definition.get("hard_negative_weighting", False)),
        scale=float(definition.get("hard_negative_scale", 1.2)),
    )

    output_directory = Path(output_directory)
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_total = 0.0
        step_count = 0
        skipped_singletons = 0
        for batch in loader:
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            class_ids = batch["class_ids"].to(device, non_blocking=True)
            if dataset.mode == "audio_only":
                logits = model(waveforms)
                loss = functional.cross_entropy(logits, class_ids)
            else:
                if waveforms.shape[0] < 2:
                    skipped_singletons += 1
                    continue
                tokens = batch["tokens"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                audio_embeddings = model.encode_audio(waveforms)
                prompt_count = int(batch["prompt_count"])
                text_embeddings = model.encode_text(tokens, attention_mask).reshape(
                    waveforms.shape[0], prompt_count, -1
                )
                if prompt_count > 1:
                    weights = batch["soft_weights"].to(device, non_blocking=True)
                    text_embeddings = functional.normalize(
                        (text_embeddings * weights.unsqueeze(-1)).sum(dim=1), dim=-1
                    )
                else:
                    text_embeddings = text_embeddings[:, 0]
                logits = (
                    model.logit_scale.exp().clamp(max=100)
                    * audio_embeddings
                    @ text_embeddings.T
                )
                loss = contrastive_loss(
                    logits,
                    class_ids=class_ids,
                    condition_ids=batch["condition_ids"].to(device, non_blocking=True),
                    hard_negative_policy=hard_negative,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += loss.detach().item()
            global_step += 1
            step_count += 1
        mean_loss = loss_total / max(step_count, 1)
        print(
            f"experiment={dataset.requested_experiment} epoch={epoch + 1} "
            f"loss={mean_loss:.5f} steps={global_step} skipped_singletons={skipped_singletons}"
        )
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                output_directory / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                metadata=metadata,
            )
    save_checkpoint(
        output_directory / "last.pt",
        model,
        optimizer,
        epochs - 1,
        global_step,
        metadata=metadata,
    )
