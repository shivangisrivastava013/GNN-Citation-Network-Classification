# Graph Neural Networks (GNN) for Citation Network Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-PyG-3B82F6?style=for-the-badge)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Implementation of **Graph Convolutional Networks (GCN)** and **GraphSAGE** architectures for semi-supervised node classification on academic citation networks (Cora / Citeseer datasets).

---

## 🌟 Architecture Overview
- **GCN (Graph Convolutional Networks):** Computes layer-wise first-order spectral graph convolutions:
  $$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
- **GraphSAGE (Sample and Aggregate):** Inductive representation learning aggregating local neighborhood feature representations.

---

## 📁 Repository Structure
```text
GNN-Citation-Network-Classification/
├── gnn_model/              # Core GNN Modules
│   ├── __init__.py
│   ├── models.py           # GCN & GraphSAGE Layer Definitions
│   └── dataset.py          # Cora Dataset & Synthetic Fallback Loader
├── demo.py                 # Quick 1-Command Training Demo
├── requirements.txt        # Package Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start
```bash
# Clone repository
git clone https://github.com/shivangisrivastava013/GNN-Citation-Network-Classification.git
cd GNN-Citation-Network-Classification

# Install dependencies
pip install -r requirements.txt

# Run demonstration
python demo.py
```

---

## 👤 Author
**Shivangi Srivastava**  
MS in Artificial Intelligence @ NJIT  
[LinkedIn Profile](https://www.linkedin.com/in/shivangisrivastava013/) | [Portfolio](https://shivangisrivastava013.github.io/shivangi-portfolio/)
