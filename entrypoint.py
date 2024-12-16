#!/usr/bin/env python3

import sys
import argparse
import torch
from temBERTure import TemBERTure


def print_usage():
    print("Usage: docker run [docker options] image [--cls|--tm] sequence")
    print("  --cls: Use TemBERTureCLS for classification")
    print("  --tm: Use TemBERTureTM for regression (uses all 3 replicas)")
    print("Example: docker run temberture --cls MEKVYGLIGFPVEH...")


def get_device():
    """Determine if CUDA is available and return appropriate device"""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return "cuda"
    else:
        print("CUDA is not available. Using CPU.")
        return "cpu"


def run_cls(sequence):
    device = get_device()
    model = TemBERTure(
        adapter_path="./temBERTure_CLS/",
        device=device,
        batch_size=1,
        task="classification",
    )
    result = model.predict(sequence)
    print(f"Result: {result}")


def run_tm(sequence):
    device = get_device()
    models = [
        TemBERTure(
            adapter_path=f"./temBERTure_TM/replica{i}/",
            device=device,
            batch_size=16,
            task="regression",
        )
        for i in range(1, 4)
    ]
    results = [model.predict(sequence) for model in models]
    print(f"Results from replicas: {results}")


def main():
    parser = argparse.ArgumentParser(description="TemBERTure prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cls", action="store_true", help="Run classification mode")
    group.add_argument(
        "--tm", action="store_true", help="Run regression mode with all replicas"
    )
    parser.add_argument("sequence", help="Protein sequence to analyze")

    try:
        args = parser.parse_args()
    except:
        print_usage()
        sys.exit(1)

    if args.cls:
        run_cls(args.sequence)
    elif args.tm:
        run_tm(args.sequence)


if __name__ == "__main__":
    main()
