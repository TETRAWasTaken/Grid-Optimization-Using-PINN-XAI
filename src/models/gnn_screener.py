import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class HierarchicalGridGNN(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_generators: int, num_lines: int):
        super(HierarchicalGridGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.line_congestion_head = nn.Linear(hidden_dim, num_lines)
        self.gen_saturation_head = nn.Linear(hidden_dim + num_lines, num_generators)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        if batch is not None:
            grid_embedding = global_mean_pool(h, batch)
        else:
            grid_embedding = torch.mean(h, dim=0, keepdim=True)
            
        line_logits = self.line_congestion_head(grid_embedding)
        line_prob = torch.sigmoid(line_logits)
        
        gen_input = torch.cat([grid_embedding, line_prob], dim=-1)
        gen_margins = 0.2 + 0.8 * torch.sigmoid(self.gen_saturation_head(gen_input))
        
        return line_prob, gen_margins