import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class LinearGCNLayer(nn.Module):
    """
    Fallback Message-Passing Graph Convolutional Layer when torch_geometric is building.
    Computes H' = D^(-1/2) A D^(-1/2) H W
    """
    def __init__(self, in_features: int, out_features: int):
        super(LinearGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        # Construct adjacency matrix from edge_index
        adj = torch.eye(num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # Degree normalization
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        h = self.linear(x)
        return torch.matmul(norm_adj, h)


class GCNNet(nn.Module):
    """
    2-Layer Graph Convolutional Network (GCN) for Semi-Supervised Node Classification.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super(GCNNet, self).__init__()
        self.dropout = dropout

        if PYG_AVAILABLE:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        else:
            self.conv1 = LinearGCNLayer(in_channels, hidden_channels)
            self.conv2 = LinearGCNLayer(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns intermediate node representations before final classification head.
        """
        x = self.conv1(x, edge_index)
        return F.relu(x)


class GraphSAGENet(nn.Module):
    """
    2-Layer GraphSAGE Network with Inductive Feature Aggregation.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super(GraphSAGENet, self).__init__()
        self.dropout = dropout

        if PYG_AVAILABLE:
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            self.conv1 = LinearGCNLayer(in_channels, hidden_channels)
            self.conv2 = LinearGCNLayer(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
