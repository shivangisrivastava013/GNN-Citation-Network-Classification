import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnn_model.datasets import load_planetoid_dataset
from gnn_model.models import get_model
from gnn_model.trainer import GNNTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GNN/MLP model on citation network dataset.")
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=["Cora", "Citeseer", "PubMed"], help="Dataset name."
    )
    parser.add_argument(
        "--model", type=str, default="gcn", choices=["mlp", "gcn", "graphsage"], help="Model architecture."
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Hidden channels.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--synthetic", action="store_true", help="Use explicit synthetic graph dataset.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save weights/results.")

    args = parser.parse_args()

    print(f"[+] Loading dataset '{args.dataset}' (synthetic={args.synthetic})...")
    data, num_features, num_classes = load_planetoid_dataset(args.dataset, use_synthetic=args.synthetic, seed=args.seed)

    model = get_model(args.model, num_features, args.hidden_channels, num_classes, args.dropout)
    trainer = GNNTrainer(
        model=model,
        data=data,
        lr=args.lr,
        weight_decay=5e-4,
        epochs=args.epochs,
        patience=args.patience,
        device="cpu",
        seed=args.seed,
    )

    print(f"[+] Training {args.model.upper()} on {args.dataset} (Seed {args.seed})...")
    weights_dir = os.path.join(args.output_dir, "weights")
    results = trainer.train(save_checkpoint_dir=weights_dir)

    print("\n[+] Training Complete!")
    print(f"[*] Best Epoch: {results['best_epoch']}")
    print(f"[*] Test Accuracy: {results['test_acc']*100:.2f}%")
    print(f"[*] Test Macro F1: {results['test_macro_f1']:.4f}")
    print(f"[*] Test Weighted F1: {results['test_weighted_f1']:.4f}")
    print(f"[*] Inference Latency: {results['inference_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
