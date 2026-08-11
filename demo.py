import os
import torch
import pandas as pd
from gnn_model.models import GCNNet, GraphSAGENet
from gnn_model.dataset import load_cora_dataset


def main():
    print("[+] Initializing GNN Citation Network Classification Engine...")
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data, num_features, num_classes = load_cora_dataset()
    data = data.to(device)

    print(f"[*] Dataset Specs | Nodes: {data.x.size(0)} | Features: {num_features} | Classes: {num_classes}")

    # Model Initialization
    model = GCNNet(in_channels=num_features, hidden_channels=64, out_channels=num_classes, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("\n[*] Training GCN for 50 epochs on citation graph...")
    history = []

    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1)
            
            train_acc = int((preds[data.train_mask] == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
            val_acc = int((preds[data.val_mask] == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
            test_acc = int((preds[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

        history.append({
            "Epoch": epoch,
            "Loss": round(loss.item(), 4),
            "Train_Acc": round(train_acc, 4),
            "Val_Acc": round(val_acc, 4),
            "Test_Acc": round(test_acc, 4)
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:2d}/50] | Loss: {loss.item():.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    # Save weights & results
    torch.save(model.state_dict(), "./weights/gcn_model.pth")
    df_history = pd.DataFrame(history)
    df_history.to_csv("./results/gnn_training_log.csv", index=False)

    print(f"\n[+] GCN Training Complete!")
    print(f"[*] Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"[*] Saved model checkpoint to ./weights/gcn_model.pth")
    print(f"[*] Saved evaluation history log to ./results/gnn_training_log.csv")


if __name__ == '__main__':
    main()
