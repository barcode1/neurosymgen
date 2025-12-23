import torch
from torch_geometric.data import HeteroData


def create_sample_hetero_kg(batch_size: int = 2, num_nodes: int = 5) -> HeteroData:
    """Create sample heterogeneous KG for testing"""
    data = HeteroData()

    # Node types
    for node_type in ["concept", "entity", "relation"]:
        data[node_type].x = torch.randn(batch_size * num_nodes, 128)

    # Edge types
    data["concept", "has", "entity"].edge_index = torch.randint(0, 10, (2, 15))
    data["entity", "relates_to", "entity"].edge_index = torch.randint(0, 10, (2, 20))

    return data