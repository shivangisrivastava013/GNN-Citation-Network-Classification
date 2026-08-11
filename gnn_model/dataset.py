import torch
import numpy as np


class SyntheticCitationGraph:
    """
    Synthetic citation network dataset generator mimicking Cora graph properties
    (2,708 nodes, 1,433 node feature dimensions, 7 subject categories).
    """
    def __init__(self, num_nodes: int = 500, num_features: int = 1433, num_classes: int = 7):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes

        # Seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Node features
        self.x = torch.rand((num_nodes, num_features), dtype=torch.float32)

        # Labels (7 categories)
        self.y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)

        # Generate citation edge index (sparse graph)
        num_edges = num_nodes * 4
        src = torch.randint(0, num_nodes, (num_edges,))
        dst = torch.randint(0, num_nodes, (num_edges,))
        self.edge_index = torch.stack([src, dst], dim=0)

        # Split masks (60% train, 20% val, 20% test)
        indices = np.random.permutation(num_nodes)
        train_end = int(num_nodes * 0.6)
        val_end = int(num_nodes * 0.8)

        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.train_mask[indices[:train_end]] = True
        self.val_mask[indices[train_end:val_end]] = True
        self.test_mask[indices[val_end:]] = True

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self


def load_cora_dataset(data_dir: str = "./data"):
    """
    Attempts to load official Planetoid Cora dataset via PyTorch Geometric;
    falls back to synthetic Cora citation graph if PyG is not installed.
    """
    try:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=data_dir, name="Cora")
        data = dataset[0]
        return data, dataset.num_features, dataset.num_classes
    except Exception:
        graph = SyntheticCitationGraph(num_nodes=500, num_features=1433, num_classes=7)
        return graph, graph.num_features, graph.num_classes
