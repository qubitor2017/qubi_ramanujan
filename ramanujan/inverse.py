from .core import SolveResult
from .linear import solve_linear
import math

def tikhonov_least_squares(A,b,lam=1e-3):
    """Solve (A^T A + lambda I)x=A^T b; dense small-problem reference."""
    if lam<0:raise ValueError('lambda must be nonnegative')
    m=len(A);n=len(A[0])
    if len(b)!=m or any(len(r)!=n for r in A):raise ValueError('dimension mismatch')
    H=[[sum(A[k][i]*A[k][j] for k in range(m))+(lam if i==j else 0) for j in range(n)] for i in range(n)]
    rhs=[sum(A[k][i]*b[k] for k in range(m)) for i in range(n)];r=solve_linear(H,rhs)
    residual=math.sqrt(sum((sum(A[i][j]*r.solution[j] for j in range(n))-b[i])**2 for i in range(m)))
    r.method='tikhonov_regularized_least_squares';r.residual=residual;r.metadata={'regularization':lam};return r

def phase_retrieval_gn(measurements, amplitudes, x0, max_iter=100, tol=1e-8):
    """Real-valued phase retrieval via damped Gauss-Newton; measurements are vectors a_k."""
    x=list(map(float,x0));m=len(measurements);n=len(x)
    if len(amplitudes)!=m or any(len(a)!=n for a in measurements):raise ValueError('dimension mismatch')
    for it in range(max_iter):
        pred=[sum(a[j]*x[j] for j in range(n)) for a in measurements];res=[pred[k]**2-amplitudes[k]**2 for k in range(m)]
        J=[[2*pred[k]*measurements[k][j] for j in range(n)] for k in range(m)]
        H=[[sum(J[k][i]*J[k][j] for k in range(m))+(1e-8 if i==j else 0) for j in range(n)] for i in range(n)]
        g=[sum(J[k][i]*res[k] for k in range(m)) for i in range(n)]
        try:dx=solve_linear(H,[-v for v in g]).solution
        except ValueError:break
        alpha=1.0;old=sum(v*v for v in res)
        while alpha>1e-8:
            z=[x[i]+alpha*dx[i] for i in range(n)];new=sum((sum(a[j]*z[j] for j in range(n))**2-amplitudes[k]**2)**2 for k,a in enumerate(measurements))
            if new<old:break
            alpha*=.5
        x=z
        if math.sqrt(new)<tol:return SolveResult(x,new,'converged','phase_retrieval_gauss_newton',it+1,residual=math.sqrt(new),metadata={'real_signal_only':True,'global_sign_ambiguity':True})
    final=sum((sum(a[j]*x[j] for j in range(n))**2-amplitudes[k]**2)**2 for k,a in enumerate(measurements))
    return SolveResult(x,final,'iteration_limit','phase_retrieval_gauss_newton',max_iter,residual=math.sqrt(final),metadata={'real_signal_only':True,'global_sign_ambiguity':True})
