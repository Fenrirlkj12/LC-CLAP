#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from src.config import CLAPConfig
from src.lc_pattern import load_method_profile
from src.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an audio-text contrastive model.")
    parser.add_argument("--manifest", required=True, help="JSONL file with audio_path and text")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--method-profile", default="profiles/lc_clap_pattern.example.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--embedding-dim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.save_every < 1:
        raise ValueError("epochs and save_every must be positive; batch_size must be at least 2")
    config = CLAPConfig(
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        embedding_dim=args.embedding_dim,
    )
    method_profile = load_method_profile(args.method_profile)
    seed = args.seed if args.seed is not None else method_profile.evaluation.seed_ids[0]
    train(
        manifest_path=args.manifest,
        output_directory=args.output_dir,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        workers=args.workers,
        device=torch.device(args.device),
        method_profile=method_profile,
        seed=seed,
        resume_checkpoint=args.resume,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
