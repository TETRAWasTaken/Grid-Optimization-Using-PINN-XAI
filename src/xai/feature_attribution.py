import numpy as np
import shap
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DispatchSHAPExplainer:
    """
    Computes Shapley additive explanations (SHAP) attributing generator dispatch
    decisions back to individual nodal loads and grid features.
    """
    def __init__(
        self, 
        pipeline: torch.nn.Module, 
        sample_graph_data, 
        background_loads: np.ndarray, 
        feature_names: list[str] = None
    ):
        self.pipeline = pipeline
        self.sample_data = sample_graph_data
        self.feature_names = feature_names or [f"Bus_{i}_Load" for i in range(background_loads.shape[1])]
        
        device = next(self.pipeline.parameters()).device
        edge_index = self.sample_data.edge_index.to(device)
        base_x = self.sample_data.x.clone().to(device)

        def model_predict(loads_np: np.ndarray) -> np.ndarray:
            self.pipeline.eval()
            dispatches = []
            with torch.no_grad():
                for row in loads_np:
                    bus_loads_t = torch.tensor(row, dtype=torch.float32, device=device)
                    x = base_x.clone()
                    x[:, 0] = bus_loads_t
                    x[:, 1] = bus_loads_t * 0.2
                    out = self.pipeline(x, edge_index, bus_loads_t)
                    dispatches.append(out.detach().cpu().numpy().flatten())
            return np.array(dispatches)

        self.predict_fn = model_predict
        background_subset = background_loads[:min(12, len(background_loads))]
        self.explainer = shap.KernelExplainer(self.predict_fn, background_subset)

    def explain_instance(self, bus_loads: np.ndarray, nsamples: int = 40):
        """
        Computes SHAP values for a single grid state instance.
        """
        if bus_loads.ndim == 1:
            bus_loads = bus_loads.reshape(1, -1)
        shap_values = self.explainer.shap_values(bus_loads, nsamples=nsamples)
        return shap_values

    def plot_attribution_bar(self, shap_values, target_gen: int = 0, save_path: str = None) -> plt.Figure:
        """
        Generates a clear bar plot of nodal load attribution for a selected generator.
        """
        if isinstance(shap_values, list):
            gen_shap = shap_values[target_gen][0]
        else:
            gen_shap = shap_values[0, :, target_gen]

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
        indices = np.arange(len(self.feature_names))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in gen_shap]

        ax.bar(indices, gen_shap, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xticks(indices)
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("SHAP Value (MW Impact)")
        ax.set_title(f"Nodal Load Marginal Attribution for Generator {target_gen} Dispatch")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
        return fig
