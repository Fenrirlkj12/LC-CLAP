#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data import HashTokenizer, load_waveform
from src.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank text labels for one audio file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = HashTokenizer(model.config.vocabulary_size, model.config.max_text_tokens)
    waveform = load_waveform(args.audio, model.config, training=False).unsqueeze(0).to(device)
    tokens, attention_mask = tokenizer.batch_encode(args.labels)
    with torch.inference_mode():
        audio_embedding = model.encode_audio(waveform)
        text_embeddings = model.encode_text(tokens.to(device), attention_mask.to(device))
        scores = (audio_embedding @ text_embeddings.T).squeeze(0)
    ordering = scores.argsort(descending=True)[: args.top_k].cpu().tolist()
    results = [
        {"label": args.labels[index], "similarity": round(float(scores[index]), 6)}
        for index in ordering
    ]
    print(json.dumps({"audio": str(args.audio), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
