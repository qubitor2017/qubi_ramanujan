from .core import SolveResult

def complex_gaussian_solve(A,b,tol=1e-12):
    n=len(A)
    if not n or len(b)!=n or any(len(r)!=n for r in A):raise ValueError('square matrix required')
    M=[[complex(v) for v in A[i]]+[complex(b[i])] for i in range(n)]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(M[i][k]))
        if abs(M[p][k])<tol:raise ValueError('singular matrix')
        M[k],M[p]=M[p],M[k];d=M[k][k];M[k]=[v/d for v in M[k]]
        for i in range(n):
            if i!=k:
                f=M[i][k];M[i]=[M[i][j]-f*M[k][j] for j in range(n+1)]
    x=[M[i][-1] for i in range(n)];return SolveResult(x,None,'solved','complex_gaussian_elimination')

def polynomial_roots(coefficients):
    """Roots of descending-order coefficients via optional NumPy."""
    try:import numpy as np
    except ImportError as e:raise ImportError('Install NumPy: pip install qubitor-ai-ramanujan[numpy]') from e
    roots=np.roots(coefficients)
    return SolveResult([complex(x) for x in roots],None,'computed','numpy_polynomial_roots',metadata={'library':'numpy'})
