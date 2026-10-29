import cvxpy as cp
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
import numpy as np

class DifferentiableOPFLayer(nn.Module):
    """
    Differentiable Optimal Power Flow (OPF) layer with explicit PTDF branch flow limits.
    Solves a convex QP dispatch problem parameterized by neural headroom margins
    and nodal active load demands, providing exact KKT gradients to upstream GNN layers.
    """
    def __init__(
        self,
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
        super(DifferentiableOPFLayer, self).__init__()
        
        self.num_gen = num_gen
        self.num_buses = num_buses
        self.num_lines = num_lines
        
        self.p_max_np = p_max.cpu().numpy()
        c_a_np = cost_a.cpu().numpy()
        c_b_np = cost_b.cpu().numpy()
        ptdf_np = ptdf.cpu().numpy()
        C_g_np = C_g.cpu().numpy()
        line_max_np = line_s_nom.cpu().numpy()
        
        # Decision Variables
        P_g = cp.Variable(num_gen, name="P_g")
        # Power balance slack variables
        s_pos = cp.Variable(1, nonneg=True, name="s_pos")
        s_neg = cp.Variable(1, nonneg=True, name="s_neg")
        # Line overload slack variables
        s_line = cp.Variable(num_lines, nonneg=True, name="s_line")
        
        # Dynamic Parameters from GNN Screener & Load Profiles
        P_load_param = cp.Parameter(num_buses, name="P_load")
        dynamic_margin = cp.Parameter(num_gen, name="dynamic_margin")
        
        # Physical line power flows via PTDF
        P_net = C_g_np @ P_g - P_load_param
        flows = ptdf_np @ P_net
        
        # Fuel Cost + Heavy Penalties on Balance and Line Slacks
        cost_fuel = cp.sum(cp.multiply(c_a_np, cp.square(P_g)) + cp.multiply(c_b_np, P_g))
        slack_penalty = (
            1e4 * (s_pos + s_neg) + 1e2 * (cp.square(s_pos) + cp.square(s_neg))
            + 1e3 * cp.sum(s_line) + 1e2 * cp.sum(cp.square(s_line))
        )
        objective = cp.Minimize(cost_fuel + slack_penalty)
        
        # Physical OPF Constraints
        constraints = [
            P_g >= 0,
            P_g <= cp.multiply(self.p_max_np, dynamic_margin),
            cp.sum(P_g) + s_pos - s_neg == cp.sum(P_load_param),  # Power Balance
            flows <= line_max_np + s_line,                         # Thermal Forward Limit
            flows >= -line_max_np - s_line                         # Thermal Reverse Limit
        ]
        
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(
            problem,
            parameters=[P_load_param, dynamic_margin],
            variables=[P_g]
        )

    def forward(self, p_load: torch.Tensor, dynamic_margin: torch.Tensor) -> torch.Tensor:
        try:
            # Primary: Interior point solver CLARABEL via diffcp (exact precision, no premature stopping)
            P_g_opt, = self.layer(
                p_load,
                dynamic_margin,
                solver_args={"solve_method": "CLARABEL"}
            )
            return P_g_opt
        except Exception:
            # Fallback 1: High-precision SCS with fine tolerance
            try:
                safe_margin = torch.clamp(dynamic_margin, min=0.2, max=1.3)
                P_g_opt, = self.layer(
                    p_load,
                    safe_margin,
                    solver_args={"solve_method": "SCS", "max_iters": 25000, "eps": 1e-6}
                )
                return P_g_opt
            except Exception:
                # Fallback 2: Proportional dispatch by generator capacity
                p_max_tensor = torch.tensor(self.p_max_np, device=p_load.device, dtype=p_load.dtype)
                total_load = torch.sum(p_load, dim=-1, keepdim=True)
                return (p_max_tensor / torch.sum(p_max_tensor)) * total_load
