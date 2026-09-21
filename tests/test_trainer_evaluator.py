import os

from gnn_model.datasets import SyntheticCitationGraph
from gnn_model.evaluator import GNNEvaluator
from gnn_model.models import GCNNet
from gnn_model.trainer import GNNTrainer


def test_gnn_evaluator():
    graph = SyntheticCitationGraph(num_nodes=100, num_features=16, num_classes=3, seed=42)
    model = GCNNet(in_channels=16, hidden_channels=8, out_channels=3)

    metrics = GNNEvaluator.evaluate(model, graph, graph.test_mask, device="cpu")
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "latency_ms" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_gnn_trainer_training_and_early_stopping(tmp_path):
    graph = SyntheticCitationGraph(num_nodes=100, num_features=16, num_classes=3, seed=42)
    model = GCNNet(in_channels=16, hidden_channels=8, out_channels=3)

    trainer = GNNTrainer(
        model=model,
        data=graph,
        lr=0.01,
        epochs=50,
        patience=5,
        device="cpu",
        seed=42,
    )

    ckpt_dir = os.path.join(tmp_path, "weights")
    results = trainer.train(save_checkpoint_dir=ckpt_dir)

    assert "best_epoch" in results
    assert "test_acc" in results
    assert "test_macro_f1" in results
    assert os.path.exists(os.path.join(ckpt_dir, "best_model.pth"))
