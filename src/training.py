from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import CLAPConfig
from .data import AudioTextDataset, HashTokenizer, collate_batch
from .lc_pattern import LCMethodProfile
from .model import CLAPModel, contrastive_loss, load_checkpoint, save_checkpoint


def train(
    manifest_path: str | Path,
    output_directory: str | Path,
    config: CLAPConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    workers: int,
    device: torch.device,
    method_profile: LCMethodProfile,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    save_every: int = 1,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = HashTokenizer(config.vocabulary_size, config.max_text_tokens)
    dataset = AudioTextDataset(
        manifest_path, config, training=True, method_profile=method_profile
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )
    model = CLAPModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    start_epoch, global_step = 0, 0
    if resume_checkpoint is not None:
        completed_epoch, global_step = load_checkpoint(
            resume_checkpoint, model, optimizer, device
        )
        start_epoch = completed_epoch + 1
    if start_epoch >= epochs:
        raise ValueError(
            "epochs must be greater than the last completed epoch in the checkpoint"
        )

    output_directory = Path(output_directory)
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_total = 0.0
        for batch in loader:
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            class_ids = batch["class_ids"].to(device, non_blocking=True)
            condition_ids = batch["condition_ids"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(
                model(waveforms, tokens, attention_mask),
                class_ids=class_ids,
                condition_ids=condition_ids,
                hard_negative_policy=method_profile.hard_negatives,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += loss.detach().item()
            global_step += 1
        mean_loss = loss_total / max(len(loader), 1)
        print(f"epoch={epoch + 1} loss={mean_loss:.5f} steps={global_step}")
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                output_directory / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                method_profile=method_profile.to_dict(),
            )
    save_checkpoint(
        output_directory / "last.pt",
        model,
        optimizer,
        epochs - 1,
        global_step,
        method_profile=method_profile.to_dict(),
    )
