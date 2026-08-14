#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from src.data import AudioTextDataset
from src.evaluation import evaluate_conditioned_retrieval, write_json
from src.lc_config import load_method_profile
from src.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one LC-CLAP run.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True, help="JSON file containing a public candidate-label list")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-profile", default="profiles/lc_clap_default.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    profile = load_method_profile(args.method_profile)
    labels = json.loads(open(args.labels, encoding="utf-8").read())
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("labels must be a JSON array of strings")
    model = load_model(args.checkpoint, device)
    dataset = AudioTextDataset(args.manifest, model.config, training=False, method_profile=profile)
    metrics = evaluate_conditioned_retrieval(model, dataset, labels, profile, device)
    write_json({"seed": args.seed, "metrics": metrics}, args.output)


if __name__ == "__main__":
    main()
