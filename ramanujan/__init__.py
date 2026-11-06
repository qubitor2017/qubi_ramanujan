"""AI_Ramanujan: classical mathematical solver library."""
from .core import SolveResult, SolverRegistry
from .linear import solve_linear, lu_factor, qr_least_squares, conjugate_gradient, power_iteration
from .roots import bisection, newton, secant, solve_nonlinear_system
from .calculus import derivative, integrate_simpson, rk4, adaptive_simpson
from .optimization import gradient_descent, newton_minimize, bfgs, coordinate_descent
from .linear_programming import solve_lp_vertex, branch_and_bound_milp
from .graphs import dijkstra, bellman_ford, floyd_warshall, kruskal_mst, max_flow_edmonds_karp, bipartite_matching
from .combinatorial import solve_knapsack, solve_tsp, maximum_independent_set, max_cut
from .number_theory import gcd, extended_gcd, mod_inverse, is_prime, prime_factors, chinese_remainder, integer_sqrt
from .complex_math import polynomial_roots, complex_gaussian_solve
from .inverse import tikhonov_least_squares, phase_retrieval_gn
from .statistics import linear_regression, covariance_matrix, pearson_correlation
from .information import entropy, cross_entropy, kl_divergence, mutual_information
__version__='0.1.0'

# Industrial backend APIs (NumPy/SciPy optional extra)
from .sdk import Ramanujan
from .industrial import (solve_linear_system, solve_sparse_linear_system, solve_lp, solve_milp,
                         solve_nonlinear_root, minimize, integrate_quad, solve_ode, tikhonov)
