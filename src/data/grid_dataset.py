import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
import pypsa
import pandapower.networks as pn

def build_ieee14_network() -> pypsa.Network:
    """
    Constructs an authentic IEEE 14-bus test feeder inside PyPSA.
    """
    pp_net = pn.case14()
    n = pypsa.Network()

    # 1. Add Buses
    for i, b in pp_net.bus.iterrows():
        n.add("Bus", f"Bus {i}", v_nom=b.vn_kv)

    # 2. Add Transmission Lines
    for i, l in pp_net.line.iterrows():
        v_nom = pp_net.bus.loc[l.from_bus, "vn_kv"]
        z_base = (v_nom ** 2) / 100.0  # 100 MVA base
        r = (l.r_ohm_per_km * l.length_km) / z_base
        x = (l.x_ohm_per_km * l.length_km) / z_base
        s_nom = float(np.sqrt(3) * v_nom * l.max_i_ka)
        n.add(
            "Line",
            f"Line {i}",
            bus0=f"Bus {l.from_bus}",
            bus1=f"Bus {l.to_bus}",
            r=r,
            x=x,
            s_nom=max(s_nom, 40.0)
        )

    # 3. Add Transformers as Lines with impedance
    for i, t in pp_net.trafo.iterrows():
        x = (t.vk_percent / 100.0) * (100.0 / t.sn_mva)
        r = (t.vkr_percent / 100.0) * (100.0 / t.sn_mva)
        n.add(
            "Line",
            f"Trafo {i}",
            bus0=f"Bus {t.hv_bus}",
            bus1=f"Bus {t.lv_bus}",
            r=max(r, 1e-4),
            x=max(x, 1e-3),
            s_nom=float(t.sn_mva)
        )

    # 4. Add Generators (1 Slack Gen at Bus 0, 4 PV Generators at Buses 1, 2, 5, 7)
    n.add("Generator", "Slack_Gen_0", bus="Bus 0", p_nom=332.4, p_min_pu=0.0, marginal_cost=20.0)
    costs = [25.0, 32.0, 38.0, 45.0]
    for i, g in pp_net.gen.iterrows():
        n.add(
            "Generator",
            f"Gen_{i}",
            bus=f"Bus {g.bus}",
            p_nom=float(g.max_p_mw if g.max_p_mw > 0 else 100.0),
            p_min_pu=0.0,
            marginal_cost=costs[i % len(costs)]
        )

    # 5. Add Base Loads
    for i, ld in pp_net.load.iterrows():
        n.add("Load", f"Load_{i}", bus=f"Bus {ld.bus}", p_set=float(ld.p_mw))

    return n

def build_ac_dc_meshed_network() -> pypsa.Network:
    """Fallback benchmark: ac_dc_meshed example network."""
    return pypsa.examples.ac_dc_meshed()


class PyPSAGridDataset(Dataset):
    """
    Parses power grid topologies (IEEE 14 / ac_dc_meshed), executes real DC-OPF
    solves using PyPSA & HiGHS, caches the labeled dataset on disk, and yields
    PyG Data objects with full physical graph attributes.
    """
    def __init__(
        self,
        benchmark: str = "ieee14",
        num_samples: int = 8760,
        cache_dir: str = "data/processed",
        force_regenerate: bool = False
    ):
        super(PyPSAGridDataset, self).__init__()
        self.benchmark = benchmark.lower()
        self.num_samples = num_samples
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"dataset_{self.benchmark}_{self.num_samples}.pt")

        if self.benchmark == "ieee14":
            self.net = build_ieee14_network()
        else:
            self.net = build_ac_dc_meshed_network()

        # Grid Dimensions
        self.num_buses = len(self.net.buses)
        self.num_generators = len(self.net.generators)
        self.num_lines = len(self.net.lines)

        # Map buses to indices
        self.bus_to_idx = {bus: idx for idx, bus in enumerate(self.net.buses.index)}
        self.gen_to_idx = {gen: idx for idx, gen in enumerate(self.net.generators.index)}
        self.line_to_idx = {line: idx for idx, line in enumerate(self.net.lines.index)}

        # Generator-to-Bus incidence mapping (C_g matrix in R^{B x G})
        self.gen_to_bus_idx = torch.tensor(
            [self.bus_to_idx[g.bus] for _, g in self.net.generators.iterrows()],
            dtype=torch.long
        )
        C_g_np = np.zeros((self.num_buses, self.num_generators))
        for g_idx, b_idx in enumerate(self.gen_to_bus_idx.numpy()):
            C_g_np[b_idx, g_idx] = 1.0
        self.C_g = torch.tensor(C_g_np, dtype=torch.float32)

        # Graph Connectivity (Edge Index)
        edges = []
        for _, line in self.net.lines.iterrows():
            u = self.bus_to_idx[line.bus0]
            v = self.bus_to_idx[line.bus1]
            edges.append([u, v])
            edges.append([v, u])
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Generator capacity & cost parameters
        self.p_max = torch.tensor(self.net.generators.p_nom.values, dtype=torch.float32)
        self.cost_b = torch.tensor(self.net.generators.marginal_cost.values, dtype=torch.float32)
        self.cost_a = torch.tensor([0.005] * self.num_generators, dtype=torch.float32)
        self.line_s_nom = torch.tensor(self.net.lines.s_nom.values, dtype=torch.float32)

        # Calculate exact PTDF Matrix
        self.net.determine_network_topology()
        sub_net = self.net.sub_networks.obj.iloc[0]
        sub_net.calculate_PTDF()
        self.ptdf = torch.tensor(sub_net.PTDF, dtype=torch.float32)

        # Load or solve dataset
        if os.path.exists(cache_file) and not force_regenerate:
            print(f"Loading precomputed OPF dataset from cache: {cache_file}")
            self.data_list = torch.load(cache_file, weights_only=False)
        else:
            print(f"Solving {self.num_samples} DC-OPF snapshots using PyPSA + HiGHS solver...")
            self.data_list = self._solve_and_cache_dataset(cache_file)

    def _solve_and_cache_dataset(self, cache_file: str) -> list[Data]:
        snapshots = pd.date_range("2024-01-01", periods=self.num_samples, freq="h")
        self.net.set_snapshots(snapshots)

        np.random.seed(42)
        # Create diurnal load shape with seasonal variations + noise
        hours = np.arange(self.num_samples)
        daily_cycle = 1.0 + 0.25 * np.sin(2 * np.pi * (hours - 8) / 24)
        seasonal_cycle = 1.0 + 0.10 * np.sin(2 * np.pi * hours / 8760)

        # Populate loads for each snapshot in PyPSA
        bus_loads_matrix = np.zeros((self.num_samples, self.num_buses))
        for _, ld in self.net.loads.iterrows():
            b_idx = self.bus_to_idx[ld.bus]
            random_noise = np.random.uniform(0.9, 1.1, size=self.num_samples)
            series = ld.p_set * daily_cycle * seasonal_cycle * random_noise
            self.net.add("Load", f"{ld.name}_dynamic", bus=ld.bus, p_set=pd.Series(series, index=snapshots))
            bus_loads_matrix[:, b_idx] += series

        # Remove static loads to avoid duplication
        static_load_names = [name for name in self.net.loads.index if not name.endswith("_dynamic")]
        for name in static_load_names:
            self.net.remove("Load", name)

        # Run multi-snapshot DC-OPF via HiGHS
        status, cond = self.net.optimize(solver_name="highs")
        print(f"PyPSA DC-OPF Finished! Status: {status}, Condition: {cond}")

        # Extract optimal dispatches & line power flows
        opt_dispatches = self.net.generators_t.p.values  # Shape: (num_samples, num_gen)
        opt_line_flows = self.net.lines_t.p0.values      # Shape: (num_samples, num_lines)

        # Generate solar and wind capacity factors
        solar_profiles = np.clip(np.sin(np.pi * ((hours % 24) - 6) / 12), 0, 1)[:, None]
        solar_profiles = np.repeat(solar_profiles, self.num_buses, axis=1) * np.random.uniform(0.8, 1.0, size=(self.num_samples, self.num_buses))
        wind_profiles = np.random.weibull(2.0, size=(self.num_samples, self.num_buses)) * 0.35
        wind_profiles = np.clip(wind_profiles, 0.05, 0.95)

        data_list = []
        for i in range(self.num_samples):
            bus_load_i = bus_loads_matrix[i]
            x_i = np.column_stack([
                bus_load_i,
                bus_load_i * 0.2,
                solar_profiles[i],
                wind_profiles[i]
            ])
            data_obj = Data(
                x=torch.tensor(x_i, dtype=torch.float32),
                edge_index=self.edge_index,
                bus_loads=torch.tensor(bus_load_i, dtype=torch.float32),
                y=torch.tensor(opt_dispatches[i], dtype=torch.float32),
                line_flows=torch.tensor(opt_line_flows[i], dtype=torch.float32)
            )
            data_list.append(data_obj)

        print(f"Saving {len(data_list)} solved OPF samples to {cache_file}...")
        torch.save(data_list, cache_file)
        return data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

def get_train_val_test_datasets(
    benchmark: str = "ieee14",
    num_samples: int = 8760,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    cache_dir: str = "data/processed"
):
    """
    Returns train, validation, and test subsets of the PyPSAGridDataset.
    """
    full_dataset = PyPSAGridDataset(benchmark=benchmark, num_samples=num_samples, cache_dir=cache_dir)
    n_total = len(full_dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    from torch.utils.data import Subset
    train_set = Subset(full_dataset, list(range(0, n_train)))
    val_set = Subset(full_dataset, list(range(n_train, n_train + n_val)))
    test_set = Subset(full_dataset, list(range(n_train + n_val, n_total)))

    return full_dataset, train_set, val_set, test_set
