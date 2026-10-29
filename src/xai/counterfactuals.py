import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class CounterfactualDispatcher:
    """
    Generates actionable 'What-If' counterfactual explanations for grid operators.
    Calculates minimal nodal load interventions to achieve target generator setpoints
    (e.g., curtailing fossil peakers or avoiding branch overloads).
    """
    def __init__(self, model_predict_fn):
        self.predict_fn = model_predict_fn

    def generate_counterfactual(
        self,
        current_loads: np.ndarray,
        target_gen_idx: int = 1,
        target_setpoint: float = 0.0,
        max_delta_pct: float = 0.25,
        reg_weight: float = 0.01
    ) -> dict:
        """
        Solves:
            min_{delta} (P_g[target_gen_idx](x0 + delta) - target_setpoint)^2 + reg_weight * ||delta||^2
        subject to:
            -max_delta_pct * x0 <= delta <= max_delta_pct * x0
        """
        x0 = np.array(current_loads, dtype=np.float64).flatten()
        bounds = [(-max_delta_pct * val, max_delta_pct * val) for val in x0]
        
        orig_dispatch = self.predict_fn(x0.reshape(1, -1))[0]

        def objective(delta):
            x_candidate = (x0 + delta).reshape(1, -1)
            pred = self.predict_fn(x_candidate)[0]
            mismatch = (pred[target_gen_idx] - target_setpoint) ** 2
            penalty = reg_weight * np.sum(delta ** 2)
            return mismatch + penalty

        init_delta = np.zeros_like(x0)
        res = minimize(objective, init_delta, bounds=bounds, method="L-BFGS-B")
        
        cf_loads = x0 + res.x
        cf_dispatch = self.predict_fn(cf_loads.reshape(1, -1))[0]

        return {
            "success": bool(res.success),
            "original_loads": x0,
            "counterfactual_loads": cf_loads,
            "delta_loads": res.x,
            "original_dispatch": orig_dispatch,
            "counterfactual_dispatch": cf_dispatch,
            "target_gen_idx": target_gen_idx,
            "target_setpoint": target_setpoint,
            "achieved_setpoint": float(cf_dispatch[target_gen_idx]),
            "max_delta_mw": float(np.max(np.abs(res.x))),
            "total_curtailment_mw": float(np.sum(res.x))
        }

    def plot_counterfactual_comparison(self, cf_result: dict, bus_names: list[str] = None, gen_names: list[str] = None, save_path: str = None) -> plt.Figure:
        """
        Creates a dual-panel plot showing:
        1. Nodal load interventions (before vs. after).
        2. Resulting generator dispatch shift.
        """
        num_buses = len(cf_result["original_loads"])
        num_gens = len(cf_result["original_dispatch"])
        b_names = bus_names or [f"B{i}" for i in range(num_buses)]
        g_names = gen_names or [f"G{i}" for i in range(num_gens)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
        
        # Panel 1: Load Adjustments
        idx_b = np.arange(num_buses)
        width = 0.35
        ax1.bar(idx_b - width/2, cf_result["original_loads"], width, label="Baseline Load", color="#7f8c8d")
        ax1.bar(idx_b + width/2, cf_result["counterfactual_loads"], width, label="Counterfactual Load", color="#e67e22")
        ax1.set_xticks(idx_b)
        ax1.set_xticklabels(b_names, rotation=45)
        ax1.set_ylabel("Active Power Demand (MW)")
        ax1.set_title("Nodal Demand Interventions (What-If)")
        ax1.legend()
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # Panel 2: Generator Dispatches
        idx_g = np.arange(num_gens)
        ax2.bar(idx_g - width/2, cf_result["original_dispatch"], width, label="Baseline Dispatch", color="#2980b9")
        ax2.bar(idx_g + width/2, cf_result["counterfactual_dispatch"], width, label="Counterfactual Dispatch", color="#27ae60")
        ax2.set_xticks(idx_g)
        ax2.set_xticklabels(g_names)
        ax2.set_ylabel("Generation Output (MW)")
        ax2.set_title(f"Target Gen {cf_result['target_gen_idx']} Curtailment Response")
        ax2.legend()
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
        return fig
