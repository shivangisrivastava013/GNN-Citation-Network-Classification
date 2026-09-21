import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class MLPNet(nn.Module):
    """
    2-Layer Multi-Layer Perceptron (MLP) baseline.
    Ignores graph edge topology and relies solely on node feature vectors.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = self.fc1(x)
        return F.relu(x)


class GCNNet(nn.Module):
    """
    2-Layer Graph Convolutional Network (GCN) for Semi-Supervised Node Classification.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return F.relu(x)


class GraphSAGENet(nn.Module):
    """
    2-Layer GraphSAGE Network with Inductive Feature Aggregation.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        return F.relu(x)


def get_model(model_name: str, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
    """
    Factory function to instantiate models by name.
    """
    name_lower = model_name.lower()
    if name_lower == "mlp":
        return MLPNet(in_channels, hidden_channels, out_channels, dropout)
    elif name_lower == "gcn":
        return GCNNet(in_channels, hidden_channels, out_channels, dropout)
    elif name_lower in ("sage", "graphsage"):
        return GraphSAGENet(in_channels, hidden_channels, out_channels, dropout)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: 'mlp', 'gcn', 'graphsage'.")
