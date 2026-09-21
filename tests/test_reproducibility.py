from gnn_model.datasets import SyntheticCitationGraph
from gnn_model.models import GCNNet
from gnn_model.trainer import GNNTrainer


def test_training_reproducibility():
    graph1 = SyntheticCitationGraph(num_nodes=100, num_features=16, num_classes=3, seed=42)
    model1 = GCNNet(in_channels=16, hidden_channels=8, out_channels=3)
    trainer1 = GNNTrainer(model=model1, data=graph1, epochs=20, seed=42)
    res1 = trainer1.train()

    graph2 = SyntheticCitationGraph(num_nodes=100, num_features=16, num_classes=3, seed=42)
    model2 = GCNNet(in_channels=16, hidden_channels=8, out_channels=3)
    trainer2 = GNNTrainer(model=model2, data=graph2, epochs=20, seed=42)
    res2 = trainer2.train()

    assert res1["test_acc"] == res2["test_acc"]
    assert res1["test_macro_f1"] == res2["test_macro_f1"]
