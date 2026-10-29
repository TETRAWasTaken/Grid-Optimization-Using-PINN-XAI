import sys
import types

# Compatibility shim for older PySINDy builds looking for pkg_resources on Python 3.10+
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

import pysindy as ps
import numpy as np

class PolicySymbolicDistiller:
    """
    Distills continuous grid state -> dispatch trajectories into interpretable
    sparse symbolic formulas using Sparse Identification of Nonlinear Dynamics (PySINDy).
    """
    def __init__(self, polynomial_degree: int = 1, threshold: float = 0.01):
        feature_library = ps.PolynomialLibrary(degree=polynomial_degree)
        optimizer = ps.STLSQ(threshold=threshold)
        self.model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
        self.equations = []

    def fit_and_export_equations(
        self, 
        grid_inputs: np.ndarray, 
        dispatch_outputs: np.ndarray,
        feature_names: list[str] = None
    ) -> list[str]:
        """
        Fits sparse linear/polynomial equations for each generator output from nodal demands.
        """
        # SINDy requires sample time t when fitting (x_dot corresponds to target dispatch)
        self.model.fit(grid_inputs, t=1.0, x_dot=dispatch_outputs, feature_names=feature_names)
        
        raw_eqs = self.model.equations(precision=4)
        self.equations = []
        for idx, eq in enumerate(raw_eqs):
            formatted = f"P_g[{idx}] = {eq}"
            self.equations.append(formatted)

        print("\n--- Distilled Symbolic Dispatch Policy Formulas ---")
        for eq in self.equations:
            print(eq)

        return self.equations

    def get_equations_dict(self) -> dict[str, str]:
        return {f"Generator_{i}": eq for i, eq in enumerate(self.equations)}
