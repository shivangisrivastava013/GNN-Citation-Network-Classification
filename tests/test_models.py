import pytest
import torch

from gnn_model.models import GCNNet, GraphSAGENet, MLPNet, get_model


@pytest.fixture
def sample_graph_data():
    x = torch.randn(100, 32)
    edge_index = torch.randint(0, 100, (2, 300))
    return x, edge_index


def test_mlp_forward(sample_graph_data):
    x, edge_index = sample_graph_data
    model = MLPNet(in_channels=32, hidden_channels=16, out_channels=5)
    out = model(x, edge_index)
    assert out.shape == (100, 5)

    emb = model.get_embeddings(x)
    assert emb.shape == (100, 16)


def test_gcn_forward(sample_graph_data):
    x, edge_index = sample_graph_data
    model = GCNNet(in_channels=32, hidden_channels=16, out_channels=5)
    out = model(x, edge_index)
    assert out.shape == (100, 5)

    emb = model.get_embeddings(x, edge_index)
    assert emb.shape == (100, 16)


def test_graphsage_forward(sample_graph_data):
    x, edge_index = sample_graph_data
    model = GraphSAGENet(in_channels=32, hidden_channels=16, out_channels=5)
    out = model(x, edge_index)
    assert out.shape == (100, 5)

    emb = model.get_embeddings(x, edge_index)
    assert emb.shape == (100, 16)


def test_get_model_factory():
    mlp = get_model("mlp", 32, 16, 5)
    assert isinstance(mlp, MLPNet)

    gcn = get_model("gcn", 32, 16, 5)
    assert isinstance(gcn, GCNNet)

    sage = get_model("graphsage", 32, 16, 5)
    assert isinstance(sage, GraphSAGENet)

    with pytest.raises(ValueError, match="Unknown model name"):
        get_model("invalid_model", 32, 16, 5)
