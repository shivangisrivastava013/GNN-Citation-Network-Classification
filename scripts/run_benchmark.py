import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnn_model.experiment import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run complete multi-seed GNN citation benchmark suite.")
    parser.add_argument(
        "--datasets", nargs="+", default=["Cora", "Citeseer"], choices=["Cora", "Citeseer", "PubMed"], help="Datasets."
    )
    parser.add_argument(
        "--models", nargs="+", default=["MLP", "GCN", "GraphSAGE"], choices=["MLP", "GCN", "GraphSAGE"], help="Models."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2026], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per run.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--synthetic", action="store_true", help="Run with explicit synthetic citation graph.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Planetoid data directory.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save benchmark results.")

    args = parser.parse_args()

    runner = BenchmarkRunner(
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        use_synthetic=args.synthetic,
        epochs=args.epochs,
        patience=args.patience,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
