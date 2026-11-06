"""Production-oriented adapters to established numerical backends.

These APIs preserve backend status and diagnostics. Install with:
    pip install qubitor-ai-ramanujan[industrial]
"""
from __future__ import annotations
from typing import Callable, Sequence
import math
import time
from .core import SolveResult, BackendUnavailableError, SolverInputError


def _scipy():
    try:
        import numpy as np
        import scipy
        return np, scipy
    except ImportError as exc:
        raise BackendUnavailableError("This operation requires NumPy and SciPy. Install qubitor-ai-ramanujan[industrial].") from exc


def _result_from_scipy(res, method, start, *, objective=None, residual=None, metadata=None):
    elapsed = time.perf_counter() - start
    status_map = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "numerical_failure"}
    code = getattr(res, "status", None)
    status = status_map.get(code, "solved" if getattr(res, "success", False) else "solver_failure")
    x = getattr(res, "x", None)
    if x is not None:
        try: x = x.tolist()
        except AttributeError: pass
    fun = objective if objective is not None else getattr(res, "fun", None)
    if fun is not None:
        try:
            import numpy as np
            candidate = np.asarray(fun)
            fun = float(candidate) if candidate.ndim == 0 else None
        except (TypeError, ValueError):
            fun = None
    nit = getattr(res, "nit", getattr(res, "mip_node_count", 0)) or 0
    try: nit = int(nit)
    except (TypeError, ValueError): nit = 0
    diag = {k: getattr(res, k) for k in ("message", "mip_gap", "mip_node_count", "mip_dual_bound", "crossover_nit") if hasattr(res, k)}
    return SolveResult(x, fun, status, method, nit, residual, metadata or {}, bool(getattr(res, "success", False)), elapsed, str(getattr(res, "message", "")), diag)


def solve_linear_system(A, b, *, method="auto", assume_a="gen", check_finite=True):
    """Dense linear solve using SciPy LAPACK-backed routines; supports complex data."""
    np, sp = _scipy(); from scipy import linalg
    a = np.asarray(A); rhs = np.asarray(b)
    if a.ndim != 2 or rhs.ndim not in (1, 2) or a.shape[0] != a.shape[1] or rhs.shape[0] != a.shape[0]:
        raise SolverInputError("A must be square and b must have matching leading dimension")
    if not np.isfinite(a).all() or not np.isfinite(rhs).all(): raise SolverInputError("A and b must contain finite values")
    t = time.perf_counter()
    try: x = linalg.solve(a, rhs, assume_a=assume_a, check_finite=check_finite)
    except linalg.LinAlgError as exc: raise SolverInputError(f"linear system could not be solved: {exc}") from exc
    residual = float(np.linalg.norm(a @ x - rhs))
    return SolveResult(x.tolist(), None, "solved", "scipy_lapack_solve", 1, residual, {"backend": "SciPy", "dtype": str(a.dtype), "shape": list(a.shape)}, True, time.perf_counter()-t, "", {"relative_residual": residual / max(float(np.linalg.norm(rhs)), 1.0)})


def solve_sparse_linear_system(A, b, *, method="spsolve", tol=1e-8, maxiter=None):
    np, sp = _scipy(); from scipy.sparse import issparse, csr_matrix
    from scipy.sparse.linalg import spsolve, cg, gmres, lsqr
    a = A if issparse(A) else csr_matrix(A); rhs = np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or rhs.shape[0] != a.shape[0]: raise SolverInputError("sparse A must be square with matching b")
    t = time.perf_counter()
    if method == "spsolve": x = spsolve(a, rhs); info = 0; iterations = 1
    elif method == "cg": x, info = cg(a, rhs, rtol=tol, atol=0.0, maxiter=maxiter); iterations = maxiter or 0
    elif method == "gmres": x, info = gmres(a, rhs, rtol=tol, atol=0.0, maxiter=maxiter); iterations = maxiter or 0
    else: raise SolverInputError("method must be spsolve, cg or gmres")
    residual = float(np.linalg.norm(a @ x-rhs))
    status = "solved" if info == 0 else ("iteration_limit" if info > 0 else "numerical_failure")
    return SolveResult(x.tolist(), None, status, f"scipy_{method}", iterations, residual, {"backend":"SciPy", "matrix_format":a.getformat(), "nnz":int(a.nnz)}, info == 0, time.perf_counter()-t, f"backend_info={info}", {"relative_residual":residual/max(float(np.linalg.norm(rhs)),1.0)})


def solve_lp(c, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method="highs", options=None):
    """Minimise c'x subject to linear constraints using SciPy/HiGHS."""
    np, sp = _scipy(); from scipy.optimize import linprog
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or not np.isfinite(c).all(): raise SolverInputError("c must be a finite one-dimensional vector")
    t = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method, options=options)
    return _result_from_scipy(res, f"scipy_linprog_{method}", t, metadata={"backend":"SciPy linprog / HiGHS", "variables":int(c.size)})


def solve_milp(c, *, integrality=None, bounds=None, constraints=None, options=None):
    """Mixed-integer linear optimisation via SciPy's HiGHS-backed MILP interface."""
    np, sp = _scipy(); from scipy.optimize import milp, Bounds, LinearConstraint
    c = np.asarray(c, dtype=float)
    if c.ndim != 1 or not np.isfinite(c).all(): raise SolverInputError("c must be a finite one-dimensional vector")
    t = time.perf_counter()
    bnd = bounds if bounds is None or isinstance(bounds, Bounds) else Bounds(*bounds)
    cons = constraints
    if cons is not None and not isinstance(cons, (list, tuple)): cons = [cons]
    res = milp(c, integrality=integrality, bounds=bnd, constraints=cons, options=options)
    return _result_from_scipy(res, "scipy_milp_highs", t, metadata={"backend":"SciPy milp / HiGHS", "variables":int(c.size)})


def solve_nonlinear_root(fun: Callable, x0, *, jac=None, method="hybr", tol=1e-9, options=None):
    np, sp = _scipy(); from scipy.optimize import root
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or not np.isfinite(x0).all(): raise SolverInputError("x0 must be a finite vector")
    t = time.perf_counter(); res = root(fun, x0, jac=jac, method=method, tol=tol, options=options)
    residual = float(np.linalg.norm(np.asarray(fun(res.x), dtype=float)))
    return _result_from_scipy(res, f"scipy_root_{method}", t, residual=residual, metadata={"backend":"SciPy", "root_residual_norm":residual})


def minimize(fun: Callable, x0, *, jac=None, hess=None, method="BFGS", bounds=None, constraints=(), tol=1e-8, options=None):
    np, sp = _scipy(); from scipy.optimize import minimize as _minimize
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1 or not np.isfinite(x0).all(): raise SolverInputError("x0 must be a finite vector")
    t = time.perf_counter(); res = _minimize(fun, x0, jac=jac, hess=hess, method=method, bounds=bounds, constraints=constraints, tol=tol, options=options)
    return _result_from_scipy(res, f"scipy_minimize_{method}", t, metadata={"backend":"SciPy", "method":method})


def integrate_quad(fun: Callable, a, b, *, epsabs=1.49e-8, epsrel=1.49e-8, limit=100):
    np, sp = _scipy(); from scipy.integrate import quad
    t=time.perf_counter(); value, error, info = quad(fun, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit, full_output=1)[:3]
    return SolveResult(float(value), None, "converged", "scipy_quad", int(info.get("neval",0)), float(error), {"backend":"SciPy", "evaluations":int(info.get("neval",0))}, True, time.perf_counter()-t, "", {"estimated_absolute_error":float(error)})


def solve_ode(fun: Callable, t_span, y0, *, method="RK45", t_eval=None, rtol=1e-6, atol=1e-9, max_step=math.inf, events=None):
    np, sp = _scipy(); from scipy.integrate import solve_ivp
    y0=np.asarray(y0, dtype=float)
    if y0.ndim != 1 or not np.isfinite(y0).all(): raise SolverInputError("y0 must be a finite vector")
    t=time.perf_counter(); res=solve_ivp(fun, t_span, y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step, events=events)
    sol={"t":res.t.tolist(), "y":res.y.tolist(), "t_events":[x.tolist() for x in res.t_events] if res.t_events is not None else []}
    return SolveResult(sol, None, "integrated" if res.success else "solver_failure", f"scipy_solve_ivp_{method}", int(res.nfev), None, {"backend":"SciPy", "nfev":int(res.nfev), "njev":int(res.njev), "nlu":int(res.nlu)}, bool(res.success), time.perf_counter()-t, str(res.message), {"status_code":int(res.status)})


def tikhonov(A, b, regularization=1e-6, *, solver="lsmr", atol=1e-8, btol=1e-8):
    np, sp = _scipy(); from scipy.sparse import issparse, eye
    from scipy.sparse.linalg import lsmr
    a=A if issparse(A) else np.asarray(A, dtype=float); rhs=np.asarray(b,dtype=float)
    if a.ndim != 2 or rhs.ndim != 1 or a.shape[0] != rhs.size: raise SolverInputError("A and b dimensions are inconsistent")
    if regularization < 0: raise SolverInputError("regularization must be nonnegative")
    t=time.perf_counter()
    if solver != "lsmr": raise SolverInputError("currently supported inverse solver is lsmr")
    # Solve augmented least squares without forming A.T @ A.
    from scipy.sparse import vstack, csr_matrix
    from scipy.sparse.linalg import lsmr as _lsmr
    aa = a if issparse(a) else csr_matrix(a)
    aug = vstack([aa, math.sqrt(regularization)*eye(aa.shape[1], format="csr")], format="csr")
    augb = np.concatenate([rhs, np.zeros(aa.shape[1])])
    out=_lsmr(aug, augb, atol=atol, btol=btol); x=out[0]
    residual=float(np.linalg.norm(aa@x-rhs))
    return SolveResult(x.tolist(), None, "converged" if out[1] in (1,2) else "iteration_limit", "scipy_lsmr_tikhonov", int(out[2]), residual, {"backend":"SciPy LSMR", "regularization":regularization, "istop":int(out[1]), "condition_estimate":float(out[6])}, out[1] in (1,2), time.perf_counter()-t, "", {"augmented_residual_norm":float(out[3])})
