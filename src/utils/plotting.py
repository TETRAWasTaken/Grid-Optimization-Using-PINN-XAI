import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_dispatch_comparison(
    predicted_dispatches: np.ndarray,
    ground_truth_dispatches: np.ndarray,
    hours: int = 24,
    gen_names: list[str] = None,
    save_path: str = "outputs/figures/dispatch_schedule_comparison.png"
) -> plt.Figure:
    """
    Plots a multi-generator dispatch schedule over time comparing PINN model against ground truth OPF.
    """
    num_gens = predicted_dispatches.shape[1]
    g_names = gen_names or [f"Gen {i}" for i in range(num_gens)]
    
    fig, axes = plt.subplots(num_gens, 1, figsize=(10, 2.5 * num_gens), sharex=True, dpi=150)
    if num_gens == 1:
        axes = [axes]

    t = np.arange(min(hours, len(predicted_dispatches)))
    for g in range(num_gens):
        ax = axes[g]
        ax.plot(t, ground_truth_dispatches[:len(t), g], label="Ground Truth (HiGHS OPF)", color="black", linestyle="--", linewidth=1.5)
        ax.plot(t, predicted_dispatches[:len(t), g], label="PINN-GNN Prediction", color="#e74c3c", linewidth=2.0)
        ax.set_ylabel(f"{g_names[g]} (MW)")
        ax.grid(True, linestyle=":", alpha=0.6)
        if g == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time Horizon (Hours)")
    fig.suptitle("24-Hour Generator Dispatch Trajectory: PINN-GNN vs. True DC-OPF", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig

def plot_training_history(history: dict, save_path: str = "outputs/figures/training_loss_curves.png") -> plt.Figure:
    """
    Plots training and validation loss progression over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], marker="o", label="Train Loss", color="#2980b9")
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        ax1.plot(epochs, history["val_loss"], marker="s", label="Val Loss", color="#e67e22")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Composite PINN Loss")
    ax1.set_title("Training & Validation Convergence")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    if "val_balance_err" in history and len(history["val_balance_err"]) == len(epochs):
        ax2.plot(epochs, history["val_balance_err"], marker="^", label="Power Balance Error (MW)", color="#27ae60")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Imbalance (MW)")
        ax2.set_title("Physics Constraint Adherence")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
