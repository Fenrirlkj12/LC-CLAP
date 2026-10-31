#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch

from src.data import TaskAudioDataset
from src.evaluation import evaluate_task, write_json
from src.model import load_model
from src.task_config import load_task_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one LC-CLAP run.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-config", required=True, help="Task JSON used for B0-B7 evaluation")
    parser.add_argument("--experiment", default=None, help="Override the checkpoint experiment name")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    task_config = load_task_config(args.task_config)
    metadata = getattr(model, "checkpoint_metadata", {})
    requested_experiment = args.experiment or metadata.get("experiment")
    experiment = args.experiment or metadata.get("base_experiment") or metadata.get("experiment")
    if not isinstance(experiment, str):
        raise ValueError("--experiment is required when checkpoint metadata has no task experiment")
    if task_config.experiment(experiment).get("mode") == "protocol_only":
        experiment = str(task_config.experiment(experiment)["base_experiment"])
    dataset = TaskAudioDataset(args.manifest, model.config, task_config, experiment, training=False)
    metrics = evaluate_task(model, dataset, task_config, experiment, device)
    seed = args.seed if args.seed is not None else metadata.get("seed", 3408)
    report = {"seed": int(seed), "experiment": requested_experiment or experiment, "metrics": metrics}
    write_json(report, args.output)


if __name__ == "__main__":
    main()
