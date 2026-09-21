import pytest

from gnn_model.datasets import SyntheticCitationGraph, load_planetoid_dataset


def test_invalid_dataset_name_raises_value_error():
    with pytest.raises(ValueError, match="is not supported"):
        load_planetoid_dataset("InvalidDatasetName")


def test_synthetic_citation_graph_properties():
    graph = SyntheticCitationGraph(num_nodes=200, num_features=64, num_classes=5, seed=42)
    assert graph.x.shape == (200, 64)
    assert graph.y.shape == (200,)
    assert graph.edge_index.shape[0] == 2
    assert graph.train_mask.sum().item() == 120  # 60% of 200
    assert graph.val_mask.sum().item() == 40  # 20% of 200
    assert graph.test_mask.sum().item() == 40  # 20% of 200
    assert graph.is_synthetic is True


def test_load_planetoid_synthetic_mode():
    data, num_features, num_classes = load_planetoid_dataset("Cora", use_synthetic=True)
    assert num_features == 1433
    assert num_classes == 7
    assert data.is_synthetic is True
