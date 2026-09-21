import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from gnn_model.evaluator import GNNEvaluator


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GNNTrainer:
    """
    Trainer managing model training on train_mask, early stopping on val_mask,
    checkpoint selection, and single final evaluation on test_mask.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        patience: int = 20,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.seed = seed
        set_seed(seed)

        self.device = torch.device(device)
        self.data = data.to(self.device)
        self.model = model.to(self.device)
        self.epochs = epochs
        self.patience = patience

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, save_checkpoint_dir: str | None = None):
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        patience_counter = 0

        history = []
        start_train_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            # Validation evaluation (NO test set evaluation here)
            val_metrics = GNNEvaluator.evaluate(self.model, self.data, self.data.val_mask, self.device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            # Early stopping based on validation loss / accuracy
            if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        training_time_s = time.perf_counter() - start_train_time

        # Restore best checkpoint
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if save_checkpoint_dir:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(save_checkpoint_dir, "best_model.pth")
            torch.save(best_model_state, ckpt_path)

        # Single final test evaluation ONLY after best checkpoint is restored
        test_metrics = GNNEvaluator.evaluate(self.model, self.data, self.data.test_mask, self.device)

        return {
            "best_epoch": best_epoch,
            "training_time_s": float(training_time_s),
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc),
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "inference_latency_ms": test_metrics["latency_ms"],
            "history": history,
        }
