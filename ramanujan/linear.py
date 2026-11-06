from .core import SolveResult
import math

def _check(A,b):
    n=len(A)
    if not n or len(b)!=n or any(len(r)!=len(A[0]) for r in A): raise ValueError('invalid matrix dimensions')
    if len(A[0])!=n: raise ValueError('A must be square')
    return n

def solve_linear(A,b,tol=1e-12):
    n=_check(A,b); M=[list(map(float,A[i]))+[float(b[i])] for i in range(n)]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(M[i][k]))
        if abs(M[p][k])<tol: raise ValueError('singular or ill-conditioned matrix')
        M[k],M[p]=M[p],M[k]; d=M[k][k]; M[k]=[v/d for v in M[k]]
        for i in range(n):
            if i!=k:
                f=M[i][k]
                for j in range(k,n+1): M[i][j]-=f*M[k][j]
    x=[M[i][-1] for i in range(n)]; res=math.sqrt(sum((sum(A[i][j]*x[j] for j in range(n))-b[i])**2 for i in range(n)))
    return SolveResult(x,None,'solved','gaussian_elimination',residual=res)

def lu_factor(A,tol=1e-12):
    n=len(A)
    if not n or any(len(r)!=n for r in A): raise ValueError('A must be square')
    U=[list(map(float,r)) for r in A]; L=[[float(i==j) for j in range(n)] for i in range(n)]; P=list(range(n))
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(U[i][k]))
        if abs(U[p][k])<tol: raise ValueError('singular matrix')
        U[k],U[p]=U[p],U[k]; P[k],P[p]=P[p],P[k]
        L[k][:k],L[p][:k]=L[p][:k],L[k][:k]
        for i in range(k+1,n):
            L[i][k]=U[i][k]/U[k][k]
            for j in range(k,n): U[i][j]-=L[i][k]*U[k][j]
    return L,U,P

def qr_least_squares(A,b,tol=1e-12):
    m=len(A); n=len(A[0]) if m else 0
    if len(b)!=m or not n or any(len(r)!=n for r in A) or m<n: raise ValueError('requires m>=n and consistent dimensions')
    cols=[[float(A[i][j]) for i in range(m)] for j in range(n)]; Q=[]; R=[[0.0]*n for _ in range(n)]
    for j,v in enumerate(cols):
        u=v[:]
        for i,q in enumerate(Q):
            R[i][j]=sum(q[k]*v[k] for k in range(m))
            u=[u[k]-R[i][j]*q[k] for k in range(m)]
        norm=math.sqrt(sum(x*x for x in u))
        if norm<tol: raise ValueError('rank deficient matrix')
        R[j][j]=norm; Q.append([x/norm for x in u])
    y=[sum(Q[i][k]*b[k] for k in range(m)) for i in range(n)]
    x=[0.0]*n
    for i in range(n-1,-1,-1): x[i]=(y[i]-sum(R[i][j]*x[j] for j in range(i+1,n)))/R[i][i]
    residual=math.sqrt(sum((sum(A[i][j]*x[j] for j in range(n))-b[i])**2 for i in range(m)))
    return SolveResult(x,None,'least_squares','modified_gram_schmidt_qr',residual=residual)

def conjugate_gradient(A,b,x0=None,tol=1e-10,max_iter=None):
    n=_check(A,b); mv=lambda x:[sum(A[i][j]*x[j] for j in range(n)) for i in range(n)]
    x=list(x0) if x0 is not None else [0.0]*n; r=[b[i]-v for i,v in enumerate(mv(x))]; p=r[:]; rr=sum(v*v for v in r); limit=max_iter or 10*n
    for it in range(limit):
        if math.sqrt(rr)<=tol: return SolveResult(x,None,'converged','conjugate_gradient',it,residual=math.sqrt(rr))
        Ap=mv(p); den=sum(p[i]*Ap[i] for i in range(n))
        if den<=0: raise ValueError('CG requires a positive-definite matrix')
        alpha=rr/den; x=[x[i]+alpha*p[i] for i in range(n)]; r=[r[i]-alpha*Ap[i] for i in range(n)]; nr=sum(v*v for v in r); beta=nr/rr; p=[r[i]+beta*p[i] for i in range(n)]; rr=nr
    return SolveResult(x,None,'iteration_limit','conjugate_gradient',limit,residual=math.sqrt(rr))

def power_iteration(A,max_iter=1000,tol=1e-10):
    n=len(A)
    if not n or any(len(r)!=n for r in A): raise ValueError('A must be square')
    x=[1/math.sqrt(n)]*n; val=0.0
    for it in range(max_iter):
        y=[sum(A[i][j]*x[j] for j in range(n)) for i in range(n)]; norm=math.sqrt(sum(v*v for v in y))
        if norm==0: return SolveResult(0.0,0.0,'zero_matrix','power_iteration',it)
        y=[v/norm for v in y]; nv=sum(y[i]*sum(A[i][j]*y[j] for j in range(n)) for i in range(n))
        if max(abs(y[i]-x[i]) for i in range(n))<tol: return SolveResult(y,nv,'converged','power_iteration',it+1)
        x,val=y,nv
    return SolveResult(x,val,'iteration_limit','power_iteration',max_iter)
