import math
from .core import SolveResult

def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _norm(x):return math.sqrt(_dot(x,x))
def gradient_descent(f,grad,x0,lr=0.01,tol=1e-8,max_iter=10000,backtracking=False):
    x=list(map(float,x0))
    for it in range(max_iter):
        g=list(grad(x));gn=_norm(g)
        if gn<tol:return SolveResult(x,f(x),'converged','gradient_descent',it,residual=gn)
        step=lr
        if backtracking:
            fx=f(x)
            while step>1e-14 and f([x[i]-step*g[i] for i in range(len(x))])>fx-1e-4*step*gn*gn:step*=.5
        x=[x[i]-step*g[i] for i in range(len(x))]
    return SolveResult(x,f(x),'iteration_limit','gradient_descent',max_iter,residual=_norm(grad(x)))

def newton_minimize(f,grad,hess,x0,tol=1e-8,max_iter=200):
    from .linear import solve_linear
    x=list(map(float,x0))
    for it in range(max_iter):
        g=list(grad(x));ng=_norm(g)
        if ng<tol:return SolveResult(x,f(x),'converged','newton_minimize',it,residual=ng)
        try:d=solve_linear(hess(x),[-v for v in g]).solution
        except ValueError: d=[-v for v in g]
        alpha=1.0;fx=f(x)
        while alpha>1e-10 and f([x[i]+alpha*d[i] for i in range(len(x))])>fx:alpha*=.5
        x=[x[i]+alpha*d[i] for i in range(len(x))]
    return SolveResult(x,f(x),'iteration_limit','newton_minimize',max_iter,residual=_norm(grad(x)))

def bfgs(f,grad,x0,tol=1e-7,max_iter=2000):
    n=len(x0);x=list(map(float,x0));H=[[float(i==j) for j in range(n)] for i in range(n)];g=list(grad(x))
    for it in range(max_iter):
        if _norm(g)<tol:return SolveResult(x,f(x),'converged','bfgs',it,residual=_norm(g))
        p=[-sum(H[i][j]*g[j] for j in range(n)) for i in range(n)];alpha=1.0;fx=f(x)
        while alpha>1e-12 and f([x[i]+alpha*p[i] for i in range(n)])>fx+1e-4*alpha*_dot(g,p):alpha*=.5
        xn=[x[i]+alpha*p[i] for i in range(n)];gn=list(grad(xn));s=[xn[i]-x[i] for i in range(n)];y=[gn[i]-g[i] for i in range(n)];ys=_dot(y,s)
        if ys>1e-14:
            rho=1/ys;I=[[float(i==j) for j in range(n)] for i in range(n)];A=[[I[i][j]-rho*s[i]*y[j] for j in range(n)] for i in range(n)];B=[[I[i][j]-rho*y[i]*s[j] for j in range(n)] for i in range(n)]
            H=[[sum(A[i][k]*sum(H[k][l]*B[l][j] for l in range(n)) for k in range(n))+rho*s[i]*s[j] for j in range(n)] for i in range(n)]
        x,g=xn,gn
    return SolveResult(x,f(x),'iteration_limit','bfgs',max_iter,residual=_norm(g))

def coordinate_descent(f,x0,step=0.1,tol=1e-8,max_iter=10000):
    x=list(map(float,x0));fx=f(x)
    for it in range(max_iter):
        changed=False
        for i in range(len(x)):
            for d in (-step,step):
                z=x[:];z[i]+=d;fz=f(z)
                if fz<fx:x,fx,changed=z,fz,True
        if not changed:
            step*=.5
            if step<tol:return SolveResult(x,fx,'converged','coordinate_descent',it)
    return SolveResult(x,fx,'iteration_limit','coordinate_descent',max_iter)
