import numpy as np
import torch

SUPPORTED_DATASETS = {"Cora", "Citeseer", "PubMed"}


class SyntheticCitationGraph:
    """
    Explicit synthetic citation graph generator for unit testing and offline smoke testing.
    Generates structured node features and graph topology.
    """

    def __init__(self, num_nodes: int = 500, num_features: int = 1433, num_classes: int = 7, seed: int = 42):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.is_synthetic = True

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate clusterable node features correlated with class labels
        self.y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)
        raw_x = torch.randn((num_nodes, num_features), dtype=torch.float32)
        # Add class-dependent offset to make features informative
        class_signals = torch.randn((num_classes, num_features), dtype=torch.float32) * 2.0
        self.x = torch.sigmoid(raw_x + class_signals[self.y])

        # Generate homophilous synthetic edge index
        num_edges = num_nodes * 4
        src_nodes = torch.randint(0, num_nodes, (num_edges,))
        # 70% edges within same class, 30% random
        same_class_mask = torch.rand(num_edges) < 0.7
        dst_nodes = torch.zeros(num_edges, dtype=torch.long)

        for i in range(num_edges):
            if same_class_mask[i]:
                src_cls = self.y[src_nodes[i]].item()
                cls_indices = (self.y == src_cls).nonzero(as_tuple=True)[0]
                dst_nodes[i] = cls_indices[torch.randint(0, len(cls_indices), (1,))[0]]
            else:
                dst_nodes[i] = torch.randint(0, num_nodes, (1,))[0]

        self.edge_index = torch.stack([src_nodes, dst_nodes], dim=0)

        # Train/Val/Test masks (60%/20%/20%)
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


def load_planetoid_dataset(name: str, root: str = "./data", use_synthetic: bool = False, seed: int = 42):
    """
    Loads Planetoid dataset (Cora, Citeseer, PubMed) via PyTorch Geometric.
    Raises ValueError for unsupported datasets.
    Raises RuntimeError if download fails without explicit use_synthetic flag.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{name}' is not supported. Choose from: {sorted(SUPPORTED_DATASETS)}")

    if use_synthetic:
        graph = SyntheticCitationGraph(num_nodes=500, num_features=1433, num_classes=7, seed=seed)
        return graph, graph.num_features, graph.num_classes

    try:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root=root, name=name)
        data = dataset[0]
        data.is_synthetic = False
        return data, dataset.num_features, dataset.num_classes
    except Exception as exc:
        raise RuntimeError(
            f"Could not load requested Planetoid dataset '{name}'. " "Use --synthetic explicitly for pipeline testing."
        ) from exc
