#!/usr/bin/env python3

import sys
import argparse
import torch
from Bio import SeqIO
import csv
from temBERTure import TemBERTure


def print_usage():
    print(
        "Usage: docker run [docker options] image [--cls|--tm] input_fasta output_tsv"
    )
    print("  --cls: Use TemBERTureCLS for classification")
    print("  --tm: Use TemBERTureTM for regression (uses all 3 replicas)")
    print("Example: docker run temberture --cls input.fasta output.tsv")


def get_device():
    """Determine if CUDA is available and return appropriate device"""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return "cuda"
    else:
        print("CUDA is not available. Using CPU.")
        return "cpu"


def read_fasta(fasta_path):
    """Read sequences from FASTA file"""
    sequences = []
    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append((record.id, str(record.seq)))
    return sequences


def run_cls(sequences, output_path):
    """Run classification for multiple sequences and save results to TSV"""
    device = get_device()
    model = TemBERTure(
        adapter_path="./temBERTure_CLS/",
        device=device,
        batch_size=1,
        task="classification",
    )

    with open(output_path, "w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(["sequence_id", "classification", "score"])

        for seq_id, sequence in sequences:
            result = model.predict(sequence)
            # Extract classification and score from result
            classification = result[0][0]  # First element of first list
            score = result[1][0]  # First element of score array
            writer.writerow([seq_id, classification, score])


def run_tm(sequences, output_path):
    """Run melting temperature prediction for multiple sequences and save results to TSV"""
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

    with open(output_path, "w", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        writer.writerow(["sequence_id", "tm1", "tm2", "tm3"])

        for seq_id, sequence in sequences:
            results = [
                model.predict(sequence)[0] for model in models
            ]  # Get first element of each prediction
            writer.writerow([seq_id] + results)


def main():
    parser = argparse.ArgumentParser(description="TemBERTure prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cls", action="store_true", help="Run classification mode")
    group.add_argument(
        "--tm", action="store_true", help="Run regression mode with all replicas"
    )
    parser.add_argument(
        "input_fasta", help="Input FASTA file containing protein sequences"
    )
    parser.add_argument("output_tsv", help="Output TSV file path")

    try:
        args = parser.parse_args()
    except:
        print_usage()
        sys.exit(1)

    # Read sequences from FASTA file
    try:
        sequences = read_fasta(args.input_fasta)
        if not sequences:
            print(f"Error: No sequences found in {args.input_fasta}")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        sys.exit(1)

    # Run appropriate analysis
    try:
        if args.cls:
            run_cls(sequences, args.output_tsv)
        elif args.tm:
            run_tm(sequences, args.output_tsv)
        print(f"Results saved to {args.output_tsv}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
