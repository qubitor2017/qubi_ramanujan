"""Unified Python entrypoint for industrial backend solvers."""
from .core import SolveResult, SolverRegistry, SolverError, SolverInputError, BackendUnavailableError
from .industrial import (solve_linear_system, solve_sparse_linear_system, solve_lp, solve_milp,
                         solve_nonlinear_root, minimize, integrate_quad, solve_ode, tikhonov)

class Ramanujan:
    """A small, stable facade over the AI_Ramanujan solver families."""
    version = "0.2.0"
    def __init__(self):
        self.registry = SolverRegistry()
        for name, fn in {
            "linear.solve": solve_linear_system,
            "sparse.solve": solve_sparse_linear_system,
            "optimisation.lp": solve_lp,
            "optimisation.milp": solve_milp,
            "nonlinear.root": solve_nonlinear_root,
            "optimisation.minimize": minimize,
            "calculus.integrate": integrate_quad,
            "scientific.ode": solve_ode,
            "inverse.tikhonov": tikhonov,
        }.items(): self.registry.register(name, fn)
    def solve(self, name, **kwargs): return self.registry.get(name)(**kwargs)
    def available_solvers(self): return self.registry.names()
