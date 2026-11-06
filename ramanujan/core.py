"""Shared result and registry contracts for AI_Ramanujan."""
from dataclasses import dataclass, field, asdict
from typing import Any, Callable
import time

@dataclass
class SolveResult:
    solution: Any
    objective: float | None = None
    status: str = "unknown"
    method: str = "unspecified"
    iterations: int = 0
    residual: float | None = None
    metadata: dict = field(default_factory=dict)
    success: bool | None = None
    runtime_seconds: float | None = None
    message: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.success is None:
            self.success = self.status.lower() in {
                "solved", "optimal", "optimal_for_enumerated_vertices", "feasible",
                "converged", "integrated", "computed", "least_squares"
            }

    def as_dict(self):
        return asdict(self)

class SolverRegistry:
    def __init__(self): self._solvers = {}
    def register(self, name: str, solver: Callable):
        if not name or name in self._solvers: raise ValueError("solver name must be unique and nonempty")
        if not callable(solver): raise TypeError("solver must be callable")
        self._solvers[name] = solver
    def get(self, name): return self._solvers[name]
    def names(self): return sorted(self._solvers)

class SolverError(RuntimeError):
    """Base exception for solver execution failures."""
class SolverInputError(SolverError, ValueError): pass
class BackendUnavailableError(SolverError, ImportError): pass

class _Timer:
    def __enter__(self): self.start = time.perf_counter(); return self
    def __exit__(self, *_): self.elapsed = time.perf_counter() - self.start
