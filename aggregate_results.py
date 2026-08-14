#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import aggregate_five_seed_reports, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate five evaluation reports.")
    parser.add_argument("--reports", nargs=5, required=True, help="Five JSON reports from evaluate.py")
    parser.add_argument("--output", default="outputs/five_seed_aggregate.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    write_json(aggregate_five_seed_reports(reports), args.output)


if __name__ == "__main__":
    main()
