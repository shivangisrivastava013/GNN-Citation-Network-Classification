import json
import os

import numpy as np
import pandas as pd

from gnn_model.datasets import load_planetoid_dataset
from gnn_model.models import get_model
from gnn_model.trainer import GNNTrainer
from gnn_model.utils import plot_benchmark_results, plot_training_curves


class BenchmarkRunner:
    """
    Executes multi-seed experiments across MLP, GCN, and GraphSAGE models on citation networks.
    Computes mean ± std dev for Accuracy, Macro F1, Weighted F1, and Latency.
    """

    def __init__(
        self,
        datasets: list | None = None,
        models: list | None = None,
        seeds: list | None = None,
        use_synthetic: bool = False,
        epochs: int = 200,
        patience: int = 20,
        data_dir: str = "./data",
        output_dir: str = "./results",
    ):
        self.datasets = datasets or ["Cora", "Citeseer", "PubMed"]
        self.models = models or ["MLP", "GCN", "GraphSAGE"]
        self.seeds = seeds or [42, 123, 456, 789, 2026]
        self.use_synthetic = use_synthetic
        self.epochs = epochs
        self.patience = patience
        self.data_dir = data_dir
        self.output_dir = output_dir

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        all_results = {}
        summary_rows = []
        sample_curves = {}

        print(f"[+] Starting GNN Benchmark Suite Across Seeds: {self.seeds}")
        print(f"[*] Datasets: {self.datasets} | Models: {self.models} | Synthetic: {self.use_synthetic}")

        for dataset_name in self.datasets:
            all_results[dataset_name] = {}

            # Load dataset template
            dataset_obj, num_features, num_classes = load_planetoid_dataset(
                dataset_name, root=self.data_dir, use_synthetic=self.use_synthetic
            )

            for model_name in self.models:
                print(f"\n[>] Evaluating {model_name} on {dataset_name}...")
                seed_metrics = []

                for seed in self.seeds:
                    # Fresh dataset instance per seed if synthetic
                    if self.use_synthetic:
                        data, num_features, num_classes = load_planetoid_dataset(
                            dataset_name, root=self.data_dir, use_synthetic=True, seed=seed
                        )
                    else:
                        data = dataset_obj

                    model = get_model(model_name, num_features, 64, num_classes, dropout=0.5)
                    trainer = GNNTrainer(
                        model=model,
                        data=data,
                        lr=0.01,
                        weight_decay=5e-4,
                        epochs=self.epochs,
                        patience=self.patience,
                        device="cpu",
                        seed=seed,
                    )

                    res = trainer.train()
                    seed_metrics.append(res)

                    if seed == self.seeds[0] and dataset_name == "Cora":
                        sample_curves[f"{model_name}"] = res["history"]

                # Calculate statistics across seeds
                accs = [m["test_acc"] for m in seed_metrics]
                macro_f1s = [m["test_macro_f1"] for m in seed_metrics]
                weighted_f1s = [m["test_weighted_f1"] for m in seed_metrics]
                train_times = [m["training_time_s"] for m in seed_metrics]
                latencies = [m["inference_latency_ms"] for m in seed_metrics]

                agg = {
                    "test_acc_mean": float(np.mean(accs)),
                    "test_acc_std": float(np.std(accs)),
                    "test_macro_f1_mean": float(np.mean(macro_f1s)),
                    "test_macro_f1_std": float(np.std(macro_f1s)),
                    "test_weighted_f1_mean": float(np.mean(weighted_f1s)),
                    "test_weighted_f1_std": float(np.std(weighted_f1s)),
                    "training_time_s_mean": float(np.mean(train_times)),
                    "inference_latency_ms_mean": float(np.mean(latencies)),
                    "raw_seed_runs": seed_metrics,
                }

                all_results[dataset_name][model_name] = agg

                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "test_acc_mean": round(agg["test_acc_mean"], 4),
                        "test_acc_std": round(agg["test_acc_std"], 4),
                        "test_macro_f1_mean": round(agg["test_macro_f1_mean"], 4),
                        "test_macro_f1_std": round(agg["test_macro_f1_std"], 4),
                        "test_weighted_f1_mean": round(agg["test_weighted_f1_mean"], 4),
                        "test_weighted_f1_std": round(agg["test_weighted_f1_std"], 4),
                        "training_time_s_mean": round(agg["training_time_s_mean"], 2),
                        "inference_latency_ms_mean": round(agg["inference_latency_ms_mean"], 2),
                    }
                )

                print(
                    f"    Result: Acc = {agg['test_acc_mean']:.4f} ± {agg['test_acc_std']:.4f} | "
                    f"Macro F1 = {agg['test_macro_f1_mean']:.4f} ± {agg['test_macro_f1_std']:.4f}"
                )

        # Save JSON & CSV
        json_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        df_summary = pd.DataFrame(summary_rows)
        csv_path = os.path.join(self.output_dir, "benchmark_summary.csv")
        df_summary.to_csv(csv_path, index=False)

        # Generate Plot Artifacts
        plot_benchmark_results(df_summary, output_dir=self.output_dir)
        if sample_curves:
            plot_training_curves(sample_curves, output_dir=self.output_dir)

        print(f"\n[+] Benchmark Complete! Artifacts saved to '{self.output_dir}'.")
        return all_results, df_summary
