#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from src.config import CLAPConfig
from src.task_config import load_task_config
from src.training import train_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one configured task experiment.")
    parser.add_argument("--manifest", required=True, help="JSONL file containing audio records")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--task-config", default=None, help="Task/experiment JSON for B0-B4")
    parser.add_argument("--experiment", default=None, help="B0-B4 or a shuffle-control recipe")
    parser.add_argument(
        "--pretrained-audio",
        default=None,
        help="B0/B1 checkpoint or explicit HTSAT model ID/path for B2-B4",
    )
    parser.add_argument(
        "--pretrained-text",
        default=None,
        help="Explicit Hugging Face RoBERTa model ID/path for B2-B4",
    )
    parser.add_argument(
        "--shuffle-scope", choices=("within_split", "global"), default="within_split"
    )
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--clip-seconds", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.task_config or not args.experiment:
        raise ValueError("--task-config and --experiment are required")
    task_config = load_task_config(args.task_config)
    experiment = task_config.experiment(args.experiment)
    base_experiment = (
        task_config.experiment(str(experiment["base_experiment"]))
        if experiment.get("mode") == "protocol_only"
        else experiment
    )
    if base_experiment.get("mode") == "audio_text" and (
        not args.pretrained_audio or not args.pretrained_text
    ):
        raise ValueError("B2-B4 require --pretrained-audio and --pretrained-text")
    if args.pretrained_text and base_experiment.get("mode") != "audio_text":
        raise ValueError("--pretrained-text is only used by B2-B4")
    defaults = task_config.training_defaults
    epochs = defaults["epochs"] if args.epochs is None else args.epochs
    batch_size = defaults["batch_size"] if args.batch_size is None else args.batch_size
    learning_rate = defaults["learning_rate"] if args.learning_rate is None else args.learning_rate
    weight_decay = defaults["weight_decay"] if args.weight_decay is None else args.weight_decay
    workers = defaults["workers"] if args.workers is None else args.workers
    save_every = defaults["save_every"] if args.save_every is None else args.save_every
    embedding_dim = defaults["embedding_dim"] if args.embedding_dim is None else args.embedding_dim
    if epochs < 1 or batch_size < 2 or learning_rate <= 0 or weight_decay < 0 or workers < 0 or save_every < 1:
        raise ValueError("epochs, batch size, learning rate, weight decay, workers, or save interval is invalid")
    config = CLAPConfig(
        sample_rate=args.sample_rate or task_config.sample_rate_hz,
        clip_seconds=args.clip_seconds or task_config.clip_seconds,
        embedding_dim=embedding_dim,
    )
    train_task(
        manifest_path=args.manifest,
        output_directory=args.output_dir,
        config=config,
        task_config=task_config,
        experiment=args.experiment,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
        device=torch.device(args.device),
        seed=args.seed if args.seed is not None else defaults["seed"],
        resume_checkpoint=args.resume,
        save_every=save_every,
        shuffle_scope=args.shuffle_scope,
        shuffle_seed=args.shuffle_seed,
        pretrained_audio=args.pretrained_audio,
        pretrained_text=args.pretrained_text,
    )


if __name__ == "__main__":
    main()
