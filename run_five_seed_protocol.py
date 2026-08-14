#!/usr/bin/env python3
"""Run the illustrative LC-CLAP-pattern train, evaluate, and aggregate flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import CLAPConfig
from src.data import AudioTextDataset
from src.evaluation import (
    aggregate_five_seed_reports,
    evaluate_conditioned_retrieval,
    write_json,
)
from src.lc_pattern import load_method_profile
from src.model import load_model
from src.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-seed LC-CLAP method-pattern protocol."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--evaluation-manifest", required=True)
    parser.add_argument("--labels", required=True, help="JSON array of public labels")
    parser.add_argument("--output-dir", default="outputs/five_seed_protocol")
    parser.add_argument("--method-profile", default="profiles/lc_clap_pattern.example.json")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_method_profile(args.method_profile)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("labels must be a JSON array of strings")
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("epochs must be positive and batch_size must be at least 2")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    reports = []
    for seed in profile.evaluation.seed_ids:
        run_dir = output_dir / f"run_{seed}"
        config = CLAPConfig()
        train(
            manifest_path=args.train_manifest,
            output_directory=run_dir,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            workers=args.workers,
            device=device,
            method_profile=profile,
            seed=seed,
        )
        model = load_model(run_dir / "last.pt", device)
        dataset = AudioTextDataset(
            args.evaluation_manifest,
            model.config,
            training=False,
            method_profile=profile,
        )
        report = {
            "seed": seed,
            "metrics": evaluate_conditioned_retrieval(
                model, dataset, labels, profile, device
            ),
        }
        reports.append(report)
        write_json(report, run_dir / "evaluation.json")
    write_json(aggregate_five_seed_reports(reports), output_dir / "aggregate.json")


if __name__ == "__main__":
    main()
