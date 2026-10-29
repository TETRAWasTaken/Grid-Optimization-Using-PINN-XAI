import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from src.utils.config import load_config
from src.data.grid_dataset import get_train_val_test_datasets
from src.models.pipeline import ExplainableOPFPipeline
from src.models.trainer import OPFTrainer
from src.xai.symbolic_distill import PolicySymbolicDistiller
from src.xai.feature_attribution import DispatchSHAPExplainer
from src.xai.counterfactuals import CounterfactualDispatcher
from src.utils.metrics import evaluate_dispatch_physics
from src.utils.plotting import plot_dispatch_comparison, plot_training_history

def main():
    # 1. Load Centralized Configuration
    config = load_config("config/grid_config.yaml")
    grid_cfg = config["grid"]
    pipe_cfg = config["pipeline"]
    xai_cfg = config["xai"]
    os.makedirs("outputs/figures", exist_ok=True)

    print(f"=== 1. Loading {grid_cfg['benchmark'].upper()} Dataset ({grid_cfg['num_samples']} Annual Snapshots) ===")
    dataset, train_set, val_set, test_set = get_train_val_test_datasets(
        benchmark=grid_cfg["benchmark"],
        num_samples=grid_cfg["num_samples"],
        train_ratio=grid_cfg["train_split"],
        val_ratio=grid_cfg["val_split"],
        cache_dir=grid_cfg.get("cache_dir", "data/processed")
    )

    batch_size = pipe_cfg["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print(f"Grid Topology: {dataset.num_buses} Buses | {dataset.num_generators} Generators | {dataset.num_lines} Transmission Lines")
    print(f"Splits -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)} | Batch Size: {batch_size}")

    print("\n=== 2. Initializing Physics-Informed Pipeline with PTDF Branch Limits ===")
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
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing training on compute device: {device.upper()}")
    if device == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    pipeline = pipeline.to(device)
    pinn_loss_cfg = pipe_cfg.get("pinn_loss", {})
    trainer = OPFTrainer(
        pipeline,
        lr=pipe_cfg["learning_rate"],
        alpha_mse=pinn_loss_cfg.get("alpha_mse", 1.0),
        beta_cost=pinn_loss_cfg.get("beta_cost", 0.0001),
        gamma_balance=pinn_loss_cfg.get("gamma_balance", 1.0),
        device=device
    )

    print("\n=== 3. Executing Physics-Informed (PINN) Decision-Focused Training ===")
    epochs = pipe_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch_idx=epoch, total_epochs=epochs)
        val_summary = trainer.evaluate(val_loader)
        print(
            f"--- Epoch {epoch}/{epochs} Complete | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_summary['val_loss']:.4f} (MSE: {val_summary['val_mse']:.4f}, Balance: {val_summary['val_balance_err']:.3f} MW) ---"
        )
        if val_summary["val_loss"] < trainer.best_val_loss:
            trainer.best_val_loss = val_summary["val_loss"]
            trainer.save_checkpoint("outputs/best_model_checkpoint.pt")

    plot_training_history(trainer.history, save_path="outputs/figures/training_loss_curves.png")

    print("\n=== 4. Out-of-Sample Test Evaluation & 24-Hour Dispatch Trajectory ===")
    pipeline.eval()
    test_preds, test_targets, test_loads = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 100:
                break
            out = pipeline(batch.x.to(device), batch.edge_index.to(device), batch.bus_loads.to(device))
            test_preds.append(out.cpu().numpy().flatten())
            test_targets.append(batch.y.cpu().numpy().flatten())
            test_loads.append(batch.bus_loads.cpu().numpy().flatten())

    test_preds_arr = np.array(test_preds)
    test_targets_arr = np.array(test_targets)
    test_loads_arr = np.array(test_loads)

    gen_names = [f"Generator {i}" for i in range(dataset.num_generators)]
    bus_names = [f"Bus {i}" for i in range(dataset.num_buses)]
    plot_dispatch_comparison(
        test_preds_arr,
        test_targets_arr,
        hours=24,
        gen_names=gen_names,
        save_path="outputs/figures/dispatch_schedule_comparison.png"
    )
    print("Saved 24-hour dispatch schedule tracking to outputs/figures/dispatch_schedule_comparison.png")

    print("\n=== 5. Extracting Data Trajectories for Symbolic Distillation (SINDy) ===")
    sym_cfg = xai_cfg["symbolic"]
    distiller = PolicySymbolicDistiller(
        polynomial_degree=sym_cfg.get("polynomial_degree", 1),
        threshold=sym_cfg.get("threshold", 0.01)
    )
    distiller.fit_and_export_equations(test_loads_arr, test_preds_arr, feature_names=bus_names)

    print("\n=== 6. Feature Attribution Analysis (SHAP) ===")
    sample_data = test_set[0]
    shap_cfg = xai_cfg["shap"]
    shap_explainer = DispatchSHAPExplainer(
        pipeline=pipeline,
        sample_graph_data=sample_data,
        background_loads=test_loads_arr[:shap_cfg.get("background_samples", 15)],
        feature_names=bus_names
    )
    
    test_instance = test_loads_arr[0]
    shap_values = shap_explainer.explain_instance(test_instance, nsamples=shap_cfg.get("eval_samples", 40))
    shap_fig = shap_explainer.plot_attribution_bar(
        shap_values,
        target_gen=0,
        save_path="outputs/figures/shap_attribution_gen0.png"
    )
    print(f"SHAP attribution bar chart saved to outputs/figures/shap_attribution_gen0.png")

    print("\n=== 7. What-If Counterfactual Explanation Generation ===")
    cf_cfg = xai_cfg["counterfactuals"]
    def predict_loads_wrapper(loads_array):
        pipeline.eval()
        results = []
        with torch.no_grad():
            for row in loads_array:
                bus_loads_t = torch.tensor(row, dtype=torch.float32, device=device)
                x = sample_data.x.clone().to(device)
                x[:, 0] = bus_loads_t
                x[:, 1] = bus_loads_t * 0.2
                out = pipeline(x, sample_data.edge_index.to(device), bus_loads_t)
                results.append(out.detach().cpu().numpy().flatten())
        return np.array(results)

    cf_dispatcher = CounterfactualDispatcher(model_predict_fn=predict_loads_wrapper)
    target_gen = 0
    current_val = float(test_preds_arr[0][target_gen])
    target_val = float(current_val * 0.85)  # 15% curtailment

    cf_result = cf_dispatcher.generate_counterfactual(
        current_loads=test_instance,
        target_gen_idx=target_gen,
        target_setpoint=target_val,
        max_delta_pct=cf_cfg.get("max_delta_pct", 0.25),
        reg_weight=cf_cfg.get("reg_weight", 0.01)
    )
    cf_dispatcher.plot_counterfactual_comparison(
        cf_result,
        bus_names=bus_names,
        gen_names=gen_names,
        save_path="outputs/figures/counterfactual_intervention.png"
    )
    print(f"Targeting Gen {target_gen} Output: {current_val:.2f} MW -> Target: {target_val:.2f} MW")
    print(f"Counterfactual Achieved: {cf_result['achieved_setpoint']:.2f} MW | Max Nodal Intervention: {cf_result['max_delta_mw']:.2f} MW")
    print("Saved counterfactual comparison plot to outputs/figures/counterfactual_intervention.png")

    print("\n=== 8. Physical Verification (Power Balance & PTDF Line Compliance) ===")
    physics_report = evaluate_dispatch_physics(
        dispatch_mw=torch.tensor(test_preds_arr[0], dtype=torch.float32),
        load_demand_mw=torch.tensor(test_loads_arr[0], dtype=torch.float32),
        p_max_mw=dataset.p_max,
        cost_a=dataset.cost_a,
        cost_b=dataset.cost_b,
        ptdf=dataset.ptdf,
        C_g=dataset.C_g,
        line_s_nom=dataset.line_s_nom
    )
    print(f"Power Balance Imbalance: {physics_report['power_balance_error_mw']:.6f} MW")
    print(f"Generator Limits Feasible: {physics_report['capacity_violations']['is_feasible']}")
    print(f"Line Flow Thermal Compliance: {physics_report['line_flow_violations']['is_flow_compliant']}")
    print(f"Overall Grid Physical Status: {'PASSED (Feasible)' if physics_report['overall_physically_valid'] else 'FAILED'}")
    print(f"Total Dispatch Operating Cost: ${physics_report['total_fuel_cost']:.2f}")

    print("\n=== Refactored Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
