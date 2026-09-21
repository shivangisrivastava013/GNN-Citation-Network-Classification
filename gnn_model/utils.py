import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_benchmark_results(df_summary: pd.DataFrame, output_dir: str = "./results"):
    """
    Plots bar charts for Test Accuracy and Macro F1 score across datasets and models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Accuracy Comparison Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_summary,
        x="dataset",
        y="test_acc_mean",
        hue="model",
        palette="viridis",
        errorbar=None,
    )
    plt.title("Model Comparison: Test Accuracy (Mean ± Std)", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(0.4, 1.0)

    # Annotate bars with mean ± std
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()

    # 2. Macro F1 Comparison Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_summary,
        x="dataset",
        y="test_macro_f1_mean",
        hue="model",
        palette="magma",
        errorbar=None,
    )
    plt.title("Model Comparison: Macro F1 Score (Mean ± Std)", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Macro F1", fontsize=12)
    plt.ylim(0.4, 1.0)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_comparison.png"), dpi=300)
    plt.close()


def plot_training_curves(curves_dict: dict, output_dir: str = "./results"):
    """
    Plots validation accuracy curves over training epochs for each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for label, history in curves_dict.items():
        df_hist = pd.DataFrame(history)
        plt.plot(df_hist["epoch"], df_hist["val_acc"], label=label, linewidth=2)

    plt.title("Validation Accuracy Curves (Seed 42)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.close()
