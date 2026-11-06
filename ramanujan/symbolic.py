"""Optional symbolic mathematics powered by SymPy."""
def _sympy():
    try:import sympy as sp;return sp
    except ImportError as e:raise ImportError('Install optional dependency: pip install qubitor-ai-ramanujan[symbolic]') from e
def simplify(expr):return _sympy().simplify(expr)
def solve_equation(expr,symbol):return _sympy().solve(expr,symbol)
def integrate(expr,symbol):return _sympy().integrate(expr,symbol)
def differentiate(expr,symbol):return _sympy().diff(expr,symbol)
def limit(expr,symbol,point):return _sympy().limit(expr,symbol,point)
