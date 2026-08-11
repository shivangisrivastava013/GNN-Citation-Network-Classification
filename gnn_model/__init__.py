"""
GNN Citation Network Node Classification Framework
Author: Shivangi Srivastava (MS in AI @ NJIT)
"""

__version__ = "1.0.0"
__author__ = "Shivangi Srivastava"

from .models import GCNNet, GraphSAGENet
from .dataset import load_cora_dataset
