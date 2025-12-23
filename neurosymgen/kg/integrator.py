import torch
import torch.nn as nn
from typing import Optional, Dict, List
import torch_geometric.nn as pyg_nn
from torch_geometric.data import HeteroData


class HeteroKGIntegrator(nn.Module):
    """
    Heterogeneous Knowledge Graph with Meta-Learning.
    Supports node types: concept, entity, relation, event.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        # Node type embeddings
        self.node_types = ["concept", "entity", "relation", "event"]
        self.type_embeddings = nn.Embedding(len(self.node_types), hidden_dim)

        # ⭐ Heterogeneous GNN (uses torch_geometric)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = pyg_nn.HeteroConv({
                ("concept", "has", "entity"): pyg_nn.GATConv(hidden_dim, hidden_dim),
                ("entity", "related_to", "entity"): pyg_nn.GATConv(hidden_dim, hidden_dim),
                ("event", "involves", "entity"): pyg_nn.GATConv(hidden_dim, hidden_dim),
                ("entity", "belongs_to", "concept"): pyg_nn.GATConv(hidden_dim, hidden_dim),
            }, aggr="sum")
            self.convs.append(conv)

        # ⭐ Meta-learning optimizer for KG updates
        self.meta_opt = torch.optim.AdamW(self.parameters(), lr=1e-4)

        self.hidden_dim = hidden_dim

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward on heterogeneous graph.

        Args:
            data: HeteroData with node_types and edge_types

        Returns:
            Updated node features dict
        """
        x_dict = {
            node_type: data[node_type].x + self.type_embeddings(
                torch.full((data[node_type].x.size(0),), i, device=data[node_type].x.device)
            )
            for i, node_type in enumerate(self.node_types)
            if node_type in data
        }

        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in data.edge_types
        }

        # Apply heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}

        return x_dict

    def update_from_feedback(self, loss: torch.Tensor, data: HeteroData):
        """Meta-learning update based on downstream task loss"""
        self.meta_opt.zero_grad()

        # Forward to get embeddings
        embeddings = self.forward(data)

        # Compute gradient w.r.t. task loss
        loss.backward(retain_graph=True)

        # Update KG parameters
        self.meta_opt.step()

        return embeddings

    def create_synthetic_kg(self, batch_size: int, num_nodes: int = 10) -> HeteroData:
        """Create synthetic KG for testing"""
        data = HeteroData()

        for i, node_type in enumerate(self.node_types):
            data[node_type].x = torch.randn(
                batch_size * num_nodes // len(self.node_types),
                self.hidden_dim
            )

            # Add edges
            if i > 0:
                src_type = self.node_types[i - 1]
                dst_type = node_type
                edge_type = (src_type, f"to_{dst_type}", dst_type)
                data[edge_type].edge_index = torch.randint(
                    0, data[src_type].x.size(0), (2, batch_size * num_nodes // 2)
                )

        return data