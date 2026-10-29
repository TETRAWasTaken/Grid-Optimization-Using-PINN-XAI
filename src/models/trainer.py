import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

class OPFTrainer:
    """
    Physics-Informed Neural Network (PINN) and Decision-Focused Trainer for Power Grid Dispatch.
    Optimizes a composite loss combining supervised imitation MSE, quadratic operational fuel cost,
    and physical power balance constraints.
    """
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        alpha_mse: float = 1.0,
        beta_cost: float = 0.0001,
        gamma_balance: float = 1.0,
        device: str = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)
        self.mse_loss = nn.MSELoss()
        
        self.alpha_mse = alpha_mse
        self.beta_cost = beta_cost
        self.gamma_balance = gamma_balance
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mse": [],
            "val_balance_err": [],
            "val_fuel_cost": []
        }
        self.best_val_loss = float("inf")

    def compute_loss(self, predicted_p_g: torch.Tensor, target_p_g: torch.Tensor, bus_loads: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # 1. Supervised Decision Loss (MSE against ground truth OPF setpoints)
        target = target_p_g.view(-1, self.model.num_gen).float()
        pred = predicted_p_g.view(-1, self.model.num_gen).float()
        l_mse = self.mse_loss(pred, target)
        
        # 2. Physics Operational Fuel Cost Loss: sum(c_a * P_g^2 + c_b * P_g)
        c_a = self.model.cost_a.to(self.device)
        c_b = self.model.cost_b.to(self.device)
        l_cost = torch.mean(torch.sum(c_a * (pred ** 2) + c_b * pred, dim=-1))
        
        # 3. Physics Power Balance Penalty: (sum(P_g) - sum(P_load))^2
        loads_sum = torch.sum(bus_loads.view(-1, self.model.num_buses), dim=-1)
        gen_sum = torch.sum(pred, dim=-1)
        l_balance = torch.mean((gen_sum - loads_sum) ** 2)
        
        total_loss = (
            self.alpha_mse * l_mse
            + self.beta_cost * l_cost
            + self.gamma_balance * l_balance
        )
        
        metrics = {
            "loss": total_loss.item(),
            "mse": l_mse.item(),
            "fuel_cost": l_cost.item(),
            "balance_err": torch.mean(torch.abs(gen_sum - loads_sum)).item()
        }
        return total_loss, metrics

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int = 1, total_epochs: int = 1) -> float:
        self.model.train()
        total_epoch_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            batch_attr = getattr(batch, "batch", None)
            
            # Forward pass: GNN screening + CVXPYLayer optimization
            predicted_setpoints = self.model(
                batch.x,
                batch.edge_index,
                batch.bus_loads,
                batch=batch_attr
            )
            
            loss, metrics = self.compute_loss(predicted_setpoints, batch.y, batch.bus_loads)
            
            # Backpropagate through KKT conditions
            loss.backward()
            self.optimizer.step()
            
            total_epoch_loss += loss.item()

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
                print(
                    f"Epoch [{epoch_idx}/{total_epochs}] | Step [{batch_idx+1}/{num_batches}] "
                    f"| Loss: {metrics['loss']:.4f} (MSE: {metrics['mse']:.4f}, Balance: {metrics['balance_err']:.3f} MW)"
                )

        avg_loss = total_epoch_loss / num_batches
        self.scheduler.step()
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        total_loss, total_mse, total_balance_err, total_cost = 0.0, 0.0, 0.0, 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_attr = getattr(batch, "batch", None)
                
                predicted_setpoints = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.bus_loads,
                    batch=batch_attr
                )
                loss, metrics = self.compute_loss(predicted_setpoints, batch.y, batch.bus_loads)
                
                total_loss += metrics["loss"]
                total_mse += metrics["mse"]
                total_balance_err += metrics["balance_err"]
                total_cost += metrics["fuel_cost"]

        summary = {
            "val_loss": total_loss / num_batches,
            "val_mse": total_mse / num_batches,
            "val_balance_err": total_balance_err / num_batches,
            "val_fuel_cost": total_cost / num_batches
        }
        self.history["val_loss"].append(summary["val_loss"])
        self.history["val_mse"].append(summary["val_mse"])
        self.history["val_balance_err"].append(summary["val_balance_err"])
        self.history["val_fuel_cost"].append(summary["val_fuel_cost"])
        return summary

    def save_checkpoint(self, path: str = "outputs/best_model_checkpoint.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss
        }, path)
        print(f"Model checkpoint saved to {path}")
