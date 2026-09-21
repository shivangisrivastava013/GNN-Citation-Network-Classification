import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gnn_model.experiment import BenchmarkRunner


def main():
    print("[+] Initializing GNN Citation Network Classification Engine...")

    synthetic_mode = "--synthetic" in sys.argv
    seeds = [42, 123, 456, 789, 2026] if not synthetic_mode else [42, 123]
    datasets = ["Cora", "Citeseer"] if not synthetic_mode else ["Cora"]

    runner = BenchmarkRunner(
        datasets=datasets,
        models=["MLP", "GCN", "GraphSAGE"],
        seeds=seeds,
        use_synthetic=synthetic_mode,
        epochs=150,
        patience=20,
        output_dir="./results",
    )
    runner.run()


if __name__ == "__main__":
    main()
