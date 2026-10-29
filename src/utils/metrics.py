import torch
import numpy as np

def compute_power_balance_error(dispatch_mw: torch.Tensor, load_demand_mw: torch.Tensor) -> float:
    """
    Computes absolute active power imbalance (MW): |sum(P_g) - sum(P_load)|
    """
    total_gen = torch.sum(dispatch_mw)
    total_load = torch.sum(load_demand_mw)
    return torch.abs(total_gen - total_load).item()

def compute_capacity_violations(dispatch_mw: torch.Tensor, p_max_mw: torch.Tensor, tol: float = 1e-2) -> dict:
    """
    Evaluates upper and lower physical generation limit violations with numerical solver tolerance.
    """
    p_max = p_max_mw.to(dispatch_mw.device)
    over_capacity = torch.clamp(dispatch_mw - p_max, min=0.0)
    under_capacity = torch.clamp(-dispatch_mw, min=0.0)
    
    total_viol = (torch.sum(over_capacity) + torch.sum(under_capacity)).item()
    
    return {
        "max_over_mw": torch.max(over_capacity).item(),
        "total_over_mw": torch.sum(over_capacity).item(),
        "max_under_mw": torch.max(under_capacity).item(),
        "total_under_mw": torch.sum(under_capacity).item(),
        "total_violation_mw": total_viol,
        "is_feasible": bool(total_viol < tol)
    }

def compute_line_flow_violations(
    dispatch_mw: torch.Tensor,
    load_demand_mw: torch.Tensor,
    ptdf: torch.Tensor,
    C_g: torch.Tensor,
    line_s_nom: torch.Tensor,
    tol: float = 1e-2
) -> dict:
    """
    Calculates PTDF-based line power flows and checks thermal limit violations.
    """
    device = dispatch_mw.device
    disp_f = dispatch_mw.to(device).float()
    load_f = load_demand_mw.to(device).float()
    ptdf_t = ptdf.to(device).float()
    C_g_t = C_g.to(device).float()
    line_limits = line_s_nom.to(device).float()
    
    # Net injection P_net = C_g * P_g - P_load
    p_net = torch.matmul(C_g_t, disp_f.unsqueeze(-1)).squeeze(-1) - load_f
    flows = torch.matmul(ptdf_t, p_net.unsqueeze(-1)).squeeze(-1)
    
    overload = torch.clamp(torch.abs(flows) - line_limits, min=0.0)
    total_overload = torch.sum(overload).item()
    max_overload = torch.max(overload).item()

    return {
        "line_flows": flows.cpu().numpy(),
        "max_overload_mw": max_overload,
        "total_overload_mw": total_overload,
        "is_flow_compliant": bool(total_overload < tol)
    }

def compute_total_generation_cost(dispatch_mw: torch.Tensor, cost_a: torch.Tensor, cost_b: torch.Tensor) -> float:
    """
    Evaluates quadratic fuel cost: sum(cost_a * P_g^2 + cost_b * P_g)
    """
    c_a = cost_a.to(dispatch_mw.device)
    c_b = cost_b.to(dispatch_mw.device)
    cost = torch.sum(c_a * (dispatch_mw ** 2) + c_b * dispatch_mw)
    return cost.item()

def evaluate_dispatch_physics(
    dispatch_mw: torch.Tensor, 
    load_demand_mw: torch.Tensor, 
    p_max_mw: torch.Tensor, 
    cost_a: torch.Tensor, 
    cost_b: torch.Tensor,
    ptdf: torch.Tensor = None,
    C_g: torch.Tensor = None,
    line_s_nom: torch.Tensor = None,
    tol: float = 1e-2
) -> dict:
    """
    Runs comprehensive physical validation for a given dispatch solution.
    """
    p_balance_err = compute_power_balance_error(dispatch_mw, load_demand_mw)
    cap_viol = compute_capacity_violations(dispatch_mw, p_max_mw, tol=tol)
    cost = compute_total_generation_cost(dispatch_mw, cost_a, cost_b)
    
    is_power_balanced = bool(p_balance_err < tol)
    overall_valid = is_power_balanced and cap_viol["is_feasible"]

    report = {
        "power_balance_error_mw": p_balance_err,
        "is_power_balanced": is_power_balanced,
        "capacity_violations": cap_viol,
        "total_fuel_cost": cost,
        "overall_physically_valid": overall_valid
    }

    if ptdf is not None and C_g is not None and line_s_nom is not None:
        flow_report = compute_line_flow_violations(dispatch_mw, load_demand_mw, ptdf, C_g, line_s_nom, tol=tol)
        report["line_flow_violations"] = flow_report
        report["overall_physically_valid"] = overall_valid and flow_report["is_flow_compliant"]

    return report
