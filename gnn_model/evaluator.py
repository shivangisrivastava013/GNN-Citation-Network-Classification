import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class GNNEvaluator:
    """
    Evaluator computing Loss, Accuracy, Macro F1, Weighted F1, and Inference Latency.
    """

    @staticmethod
    def evaluate(model: torch.nn.Module, data, mask: torch.Tensor, device: torch.device):
        model.eval()
        data = data.to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[mask], data.y[mask]).item()
            preds = out[mask].argmax(dim=1).cpu().numpy()
            targets = data.y[mask].cpu().numpy()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        acc = accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)

        return {
            "loss": float(loss),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "latency_ms": float(latency_ms),
        }
