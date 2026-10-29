import torch
import torch.nn as nn
from src.models.gnn_screener import HierarchicalGridGNN
from src.models.diff_opf_layer import DifferentiableOPFLayer

class ExplainableOPFPipeline(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_gen: int,
        num_buses: int,
        num_lines: int,
        cost_a: torch.Tensor,
        cost_b: torch.Tensor,
        p_max: torch.Tensor,
        ptdf: torch.Tensor,
        C_g: torch.Tensor,
        line_s_nom: torch.Tensor
    ):
        super(ExplainableOPFPipeline, self).__init__()
        self.num_buses = num_buses
        self.num_gen = num_gen
        self.num_lines = num_lines
        
        self.register_buffer("ptdf", ptdf)
        self.register_buffer("C_g", C_g)
        self.register_buffer("line_s_nom", line_s_nom)
        self.register_buffer("cost_a", cost_a)
        self.register_buffer("cost_b", cost_b)
        self.register_buffer("p_max", p_max)

        self.gnn = HierarchicalGridGNN(in_features, hidden_dim, num_gen, num_lines)
        self.opf_layer = DifferentiableOPFLayer(
            num_gen=num_gen,
            num_buses=num_buses,
            num_lines=num_lines,
            cost_a=cost_a,
            cost_b=cost_b,
            p_max=p_max,
            ptdf=ptdf,
            C_g=C_g,
            line_s_nom=line_s_nom
        )
        
    def forward(self, node_features, edge_index, bus_loads, batch=None, return_margins=False):
        line_probs, gen_margins = self.gnn(node_features, edge_index, batch=batch)
        p_load_batched = bus_loads.view(-1, self.num_buses)
        dispatch_setpoints = self.opf_layer(p_load_batched, gen_margins)
        
        # If single unbatched instance was passed, squeeze
        if bus_loads.ndim == 1 and bus_loads.shape[0] == self.num_buses and batch is None:
            out_dispatch = dispatch_setpoints.squeeze(0)
            if return_margins:
                return out_dispatch, gen_margins.squeeze(0), line_probs.squeeze(0)
            return out_dispatch

        if return_margins:
            return dispatch_setpoints, gen_margins, line_probs
        return dispatch_setpoints
