import sys
import time
import torch
import numpy as np
from src.utils.config import load_config
from src.data.grid_dataset import PyPSAGridDataset
from src.models.pipeline import ExplainableOPFPipeline
from src.utils.metrics import evaluate_dispatch_physics

def run_demo(snapshot_idx: int = 0):
    config = load_config("config/grid_config.yaml")
    grid_cfg = config["grid"]
    pipe_cfg = config["pipeline"]

    print("=================================================================")
    print(" Physics-Informed Neural Network (PINN-OPF) Real-Time Demo")
    print("=================================================================")

    # 1. Load Dataset & Grid Topology
    print(f"Loading {grid_cfg['benchmark'].upper()} test feeder topology...")
    dataset = PyPSAGridDataset(
        benchmark=grid_cfg["benchmark"],
        num_samples=grid_cfg["num_samples"],
        cache_dir=grid_cfg.get("cache_dir", "data/processed")
    )

    # 2. Re-create Model Architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ExplainableOPFPipeline(
        in_features=pipe_cfg["in_features"],
        hidden_dim=pipe_cfg["hidden_dim"],
        num_gen=dataset.num_generators,
        num_buses=dataset.num_buses,
        num_lines=dataset.num_lines,
        cost_a=dataset.cost_a,
        cost_b=dataset.cost_b,
        p_max=dataset.p_max,
        ptdf=dataset.ptdf,
        C_g=dataset.C_g,
        line_s_nom=dataset.line_s_nom
    ).to(device)

    ckpt_path = "outputs/best_model_checkpoint.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        pipeline.load_state_dict(state_dict)
        print(f"Loaded trained PINN model weights from {ckpt_path}")
    except Exception as e:
        print(f"Notice: Checkpoint not found ({e}). Using initialized weights for dry-run.")

    pipeline.eval()

    # 3. Benchmark Sample Forward Pass
    sample = dataset[snapshot_idx]
    x = sample.x.to(device)
    edge_index = sample.edge_index.to(device)
    bus_loads = sample.bus_loads.to(device)

    # Time neural inference
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_dispatch, gen_margins, line_probs = pipeline(x, edge_index, bus_loads, return_margins=True)
    t1 = time.perf_counter()
    neural_latency_ms = (t1 - t0) * 1000.0

    pred_dispatch_cpu = pred_dispatch.cpu()
    ground_truth_dispatch = sample.y

    # 4. Physical Compliance Report
    physics_report = evaluate_dispatch_physics(
        dispatch_mw=pred_dispatch_cpu,
        load_demand_mw=sample.bus_loads,
        p_max_mw=dataset.p_max,
        cost_a=dataset.cost_a,
        cost_b=dataset.cost_b,
        ptdf=dataset.ptdf,
        C_g=dataset.C_g,
        line_s_nom=dataset.line_s_nom
    )

    print(f"\n--- Snapshot #{snapshot_idx} Evaluation Results ---")
    print(f"Inference Latency: {neural_latency_ms:.2f} ms")
    print(f"Total Grid Demand: {torch.sum(sample.bus_loads):.2f} MW")
    print(f"Total Dispatched Generation: {torch.sum(pred_dispatch_cpu):.2f} MW")
    print(f"Power Balance Imbalance: {physics_report['power_balance_error_mw']:.4e} MW (Compliant: {physics_report['is_power_balanced']})")
    print(f"Generator Limits Feasible: {physics_report['capacity_violations']['is_feasible']}")
    print(f"Transmission Line Thermal Compliance: {physics_report['line_flow_violations']['is_flow_compliant']}")
    print(f"Operating Fuel Cost: ${physics_report['total_fuel_cost']:.2f}")

    print("\n--- Generator Dispatch Comparison ---")
    print(f"{'Generator':<12} | {'PINN-OPF (MW)':<15} | {'True OPF (MW)':<15} | {'Difference (MW)':<15}")
    print("-" * 65)
    for g in range(dataset.num_generators):
        p_pinn = pred_dispatch_cpu[g].item()
        p_true = ground_truth_dispatch[g].item()
        diff = p_pinn - p_true
        print(f"Gen {g:<8} | {p_pinn:<15.2f} | {p_true:<15.2f} | {diff:<+15.2f}")

    print("\n=================================================================")
    print(" Demo Run Finished.")

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_demo(idx)
