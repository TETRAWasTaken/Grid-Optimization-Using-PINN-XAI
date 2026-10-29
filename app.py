import sys
import types

# Compatibility shim for older packages expecting pkg_resources
try:
    import pkg_resources
except ImportError:
    pr = types.ModuleType("pkg_resources")
    class DistributionNotFound(Exception): pass
    def get_distribution(*args, **kwargs):
        class D: version = "2.0.4"
        return D()
    pr.DistributionNotFound = DistributionNotFound
    pr.get_distribution = get_distribution
    sys.modules["pkg_resources"] = pr

import time
import torch
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config
from src.data.grid_dataset import PyPSAGridDataset
from src.models.pipeline import ExplainableOPFPipeline
from src.utils.metrics import evaluate_dispatch_physics
from src.xai.feature_attribution import DispatchSHAPExplainer
from src.xai.symbolic_distill import PolicySymbolicDistiller
from src.xai.counterfactuals import CounterfactualDispatcher

# Global State
CONFIG = load_config("config/grid_config.yaml")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing Dataset for Gradio Dashboard...")
DATASET = PyPSAGridDataset(
    benchmark=CONFIG["grid"]["benchmark"],
    num_samples=CONFIG["grid"]["num_samples"],
    cache_dir=CONFIG["grid"].get("cache_dir", "data/processed")
)

# Initialize Model Pipeline
PIPELINE = ExplainableOPFPipeline(
    in_features=CONFIG["pipeline"]["in_features"],
    hidden_dim=CONFIG["pipeline"]["hidden_dim"],
    num_gen=DATASET.num_generators,
    num_buses=DATASET.num_buses,
    num_lines=DATASET.num_lines,
    cost_a=DATASET.cost_a,
    cost_b=DATASET.cost_b,
    p_max=DATASET.p_max,
    ptdf=DATASET.ptdf,
    C_g=DATASET.C_g,
    line_s_nom=DATASET.line_s_nom
).to(DEVICE)

# Try loading best checkpoint
ckpt_path = "outputs/best_model_checkpoint.pt"
try:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    PIPELINE.load_state_dict(state_dict)
    print(f"Loaded model checkpoint from {ckpt_path}")
except Exception as e:
    print(f"Running with current model weights ({e})")

PIPELINE.eval()

# Precompute sample background loads for SHAP
BG_LOADS = np.array([DATASET[i].bus_loads.numpy() for i in range(min(15, len(DATASET)))])
BUS_NAMES = [f"Bus {i}" for i in range(DATASET.num_buses)]
GEN_NAMES = [f"Gen {i}" for i in range(DATASET.num_generators)]

# Fit SINDy once on a sample subset for the dashboard
sindy_inputs = np.array([DATASET[i].bus_loads.numpy() for i in range(min(100, len(DATASET)))])
sindy_outputs = np.array([DATASET[i].y.numpy() for i in range(min(100, len(DATASET)))])
DISTILLER = PolicySymbolicDistiller(polynomial_degree=1, threshold=0.01)
DISTILLED_EQS = DISTILLER.fit_and_export_equations(sindy_inputs, sindy_outputs, feature_names=BUS_NAMES)


def simulate_dispatch(snapshot_idx: int, load_scale: float):
    sample = DATASET[int(snapshot_idx)]
    raw_loads = sample.bus_loads.clone() * float(load_scale)
    
    x = sample.x.clone().to(DEVICE)
    x[:, 0] = raw_loads.to(DEVICE)
    x[:, 1] = (raw_loads * 0.2).to(DEVICE)
    edge_index = sample.edge_index.to(DEVICE)
    bus_loads_t = raw_loads.to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_dispatch, gen_margins, line_probs = PIPELINE(x, edge_index, bus_loads_t, return_margins=True)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    pred_dispatch_cpu = pred_dispatch.cpu()
    
    physics_report = evaluate_dispatch_physics(
        dispatch_mw=pred_dispatch_cpu,
        load_demand_mw=raw_loads,
        p_max_mw=DATASET.p_max,
        cost_a=DATASET.cost_a,
        cost_b=DATASET.cost_b,
        ptdf=DATASET.ptdf,
        C_g=DATASET.C_g,
        line_s_nom=DATASET.line_s_nom
    )

    # 1. Dispatch Comparison Table
    df_dispatch = pd.DataFrame({
        "Generator": GEN_NAMES,
        "Capacity (MW)": DATASET.p_max.numpy().round(1),
        "PINN Dispatch (MW)": pred_dispatch_cpu.numpy().round(2),
        "True OPF (MW)": sample.y.numpy().round(2),
        "Headroom Margin": gen_margins.cpu().numpy().round(3)
    })

    # 2. Key Metrics Summary
    metrics_md = f"""
    ### ⚡ Physical Verification & Performance
    - **Inference Latency**: `{latency_ms:.2f} ms` (Classical OPF: ~150–400 ms)
    - **Total Active Demand**: `{torch.sum(raw_loads):.2f} MW`
    - **Total Dispatched Power**: `{torch.sum(pred_dispatch_cpu):.2f} MW`
    - **Power Balance Error**: `{physics_report['power_balance_error_mw']:.4e} MW` {'✅' if physics_report['is_power_balanced'] else '❌'}
    - **Generator Limits Compliant**: {'✅ YES' if physics_report['capacity_violations']['is_feasible'] else '❌ VIOLATION'}
    - **Line Thermal Compliance**: {'✅ YES' if physics_report['line_flow_violations']['is_flow_compliant'] else '❌ OVERLOAD'}
    - **Total Operating Fuel Cost**: `${physics_report['total_fuel_cost']:.2f}`
    """

    # 3. Line Power Flow Chart
    fig_flow, ax_f = plt.subplots(figsize=(10, 3.8), dpi=130)
    line_flows = physics_report["line_flow_violations"]["line_flows"]
    line_limits = DATASET.line_s_nom.numpy()
    line_indices = np.arange(len(line_flows))

    ax_f.bar(line_indices, np.abs(line_flows), color="#3498db", label="|Active Power Flow| (MW)", alpha=0.85)
    ax_f.step(line_indices, line_limits, where="mid", color="#e74c3c", linestyle="--", linewidth=1.5, label="Thermal Limit (MVA)")
    ax_f.set_xticks(line_indices)
    ax_f.set_xticklabels([f"L{i}" for i in line_indices], rotation=45)
    ax_f.set_ylabel("Power Flow (MW)")
    ax_f.set_title("Transmission Branch Power Flows vs. Thermal Limits")
    ax_f.legend(loc="upper right")
    ax_f.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    return df_dispatch, metrics_md, fig_flow


def compute_shap_explanation(snapshot_idx: int, target_gen_idx: int):
    sample = DATASET[int(snapshot_idx)]
    shap_explainer = DispatchSHAPExplainer(
        pipeline=PIPELINE,
        sample_graph_data=sample,
        background_loads=BG_LOADS,
        feature_names=BUS_NAMES
    )
    test_instance = sample.bus_loads.numpy()
    shap_values = shap_explainer.explain_instance(test_instance, nsamples=35)
    fig_shap = shap_explainer.plot_attribution_bar(shap_values, target_gen=int(target_gen_idx))
    return fig_shap


def run_counterfactual_explorer(snapshot_idx: int, target_gen_idx: int, curtailment_pct: float):
    sample = DATASET[int(snapshot_idx)]
    target_gen = int(target_gen_idx)
    
    def predict_loads_wrapper(loads_array):
        results = []
        with torch.no_grad():
            for row in loads_array:
                bus_loads_t = torch.tensor(row, dtype=torch.float32, device=DEVICE)
                x = sample.x.clone().to(DEVICE)
                x[:, 0] = bus_loads_t
                x[:, 1] = bus_loads_t * 0.2
                out = PIPELINE(x, sample.edge_index.to(DEVICE), bus_loads_t)
                results.append(out.detach().cpu().numpy().flatten())
        return np.array(results)

    cf_dispatcher = CounterfactualDispatcher(model_predict_fn=predict_loads_wrapper)
    current_instance = sample.bus_loads.numpy()
    orig_pred = predict_loads_wrapper(current_instance.reshape(1, -1))[0]
    curr_val = orig_pred[target_gen]
    target_val = curr_val * (1.0 - (curtailment_pct / 100.0))

    cf_res = cf_dispatcher.generate_counterfactual(
        current_loads=current_instance,
        target_gen_idx=target_gen,
        target_setpoint=target_val,
        max_delta_pct=0.25,
        reg_weight=0.01
    )

    fig_cf = cf_dispatcher.plot_counterfactual_comparison(
        cf_res,
        bus_names=BUS_NAMES,
        gen_names=GEN_NAMES
    )

    report_md = f"""
    ### 🎯 Counterfactual Intervention Report
    - **Target Generator**: `{GEN_NAMES[target_gen]}`
    - **Baseline Dispatch**: `{curr_val:.2f} MW`
    - **Target Desired Output**: `{target_val:.2f} MW` (Curtailment: `{curtailment_pct}%`)
    - **Achieved Setpoint**: `{cf_res['achieved_setpoint']:.2f} MW`
    - **Max Nodal Load Shift Required**: `{cf_res['max_delta_mw']:.2f} MW`
    - **Optimization Converged**: `{'✅ YES' if cf_res['success'] else '⚠️ Partial'}`
    """
    return fig_cf, report_md


# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌐 PINN-XAI: Physics-Informed Neural Network & Explainable AI for Power Grid Operations
        ### Multi-Task Real-Time Optimal Power Flow (OPF) with GNN Screening, CVXPY Differentiable Physics, and Decision Explainability
        ---
        """
    )

    with gr.Tabs():
        # TAB 1: Real-Time Dispatch Simulator
        with gr.Tab("⚡ Real-Time Dispatch Simulator"):
            with gr.Row():
                with gr.Column(scale=1):
                    snapshot_slider = gr.Slider(0, len(DATASET) - 1, value=0, step=1, label="Snapshot Index (Hourly Sample)")
                    load_scale_slider = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Dynamic Grid Demand Scaling (Stress Test)")
                    simulate_btn = gr.Button("⚡ Run PINN-OPF Inference", variant="primary")
                    metrics_output = gr.Markdown()

                with gr.Column(scale=2):
                    dispatch_table = gr.Dataframe(label="Generator Dispatch & Capacity Allocations")
                    flow_plot = gr.Plot(label="Transmission Line Thermal Loading Profile")

            simulate_btn.click(
                fn=simulate_dispatch,
                inputs=[snapshot_slider, load_scale_slider],
                outputs=[dispatch_table, metrics_output, flow_plot]
            )

        # TAB 2: SHAP Feature Attribution
        with gr.Tab("🔍 Feature Attribution (SHAP)"):
            gr.Markdown("### Nodal Load Attribution: Which bus demands drive specific generator setpoints?")
            with gr.Row():
                with gr.Column(scale=1):
                    shap_snapshot = gr.Slider(0, len(DATASET) - 1, value=0, step=1, label="Snapshot Index")
                    shap_gen_choice = gr.Dropdown(choices=list(range(DATASET.num_generators)), value=0, label="Target Generator")
                    shap_btn = gr.Button("🔍 Compute SHAP Values", variant="primary")
                with gr.Column(scale=2):
                    shap_plot = gr.Plot(label="Shapley Marginal Attribution Chart")

            shap_btn.click(
                fn=compute_shap_explanation,
                inputs=[shap_snapshot, shap_gen_choice],
                outputs=[shap_plot]
            )

        # TAB 3: What-If Counterfactual Explorer
        with gr.Tab("🎯 What-If Counterfactual Explorer"):
            gr.Markdown("### Actionable Counterfactuals: Minimal demand shifts required to achieve target generator curtailments.")
            with gr.Row():
                with gr.Column(scale=1):
                    cf_snapshot = gr.Slider(0, len(DATASET) - 1, value=0, step=1, label="Snapshot Index")
                    cf_gen_choice = gr.Dropdown(choices=list(range(DATASET.num_generators)), value=0, label="Target Generator to Curtail")
                    cf_curtail_pct = gr.Slider(5, 50, value=20, step=5, label="Desired Curtailment Percentage (%)")
                    cf_btn = gr.Button("🎯 Solve Counterfactual Intervention", variant="primary")
                    cf_report = gr.Markdown()
                with gr.Column(scale=2):
                    cf_plot = gr.Plot(label="Demand Intervention & Dispatch Response")

            cf_btn.click(
                fn=run_counterfactual_explorer,
                inputs=[cf_snapshot, cf_gen_choice, cf_curtail_pct],
                outputs=[cf_plot, cf_report]
            )

        # TAB 4: Symbolic Policy Formulas (SINDy)
        with gr.Tab("📐 Distilled Symbolic Formulas (SINDy)"):
            gr.Markdown("### Sparse Analytical Dispatch Policies: Distilling GNN-OPF into closed-form equations for operators.")
            eq_text = "\n\n".join([f"**{GEN_NAMES[i]} Policy**:\n```\n{eq}\n```" for i, eq in enumerate(DISTILLED_EQS)])
            gr.Markdown(eq_text)

if __name__ == "__main__":
    demo.launch(server_name=CONFIG.get("dashboard", {}).get("host", "127.0.0.1"), server_port=CONFIG.get("dashboard", {}).get("port", 7860))
