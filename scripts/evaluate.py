import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnn_model.datasets import load_planetoid_dataset
from gnn_model.evaluator import GNNEvaluator
from gnn_model.models import get_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved GNN/MLP checkpoint.")
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=["Cora", "Citeseer", "PubMed"], help="Dataset name."
    )
    parser.add_argument(
        "--model", type=str, default="gcn", choices=["mlp", "gcn", "graphsage"], help="Model architecture."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Hidden channels.")
    parser.add_argument("--synthetic", action="store_true", help="Use explicit synthetic graph dataset.")

    args = parser.parse_args()

    data, num_features, num_classes = load_planetoid_dataset(args.dataset, use_synthetic=args.synthetic)
    model = get_model(args.model, num_features, args.hidden_channels, num_classes)

    if not os.path.exists(args.checkpoint):
        print(f"[!] Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    metrics = GNNEvaluator.evaluate(model, data, data.test_mask, torch.device("cpu"))

    print(f"\n[+] Evaluation Results for {args.model.upper()} on {args.dataset}:")
    print(f"[*] Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"[*] Test Macro F1: {metrics['macro_f1']:.4f}")
    print(f"[*] Test Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"[*] Latency: {metrics['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
