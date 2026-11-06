from .core import SolveResult
import math

def bisection(f,a,b,tol=1e-10,max_iter=200):
    a=float(a);b=float(b);fa=f(a);fb=f(b)
    if fa*fb>0: raise ValueError('f(a) and f(b) must have opposite signs')
    for i in range(max_iter):
        c=(a+b)/2;fc=f(c)
        if abs(fc)<=tol or abs(b-a)<=tol: return SolveResult(c,fc,'converged','bisection',i+1,residual=abs(fc))
        if fa*fc<=0: b,fb=c,fc
        else: a,fa=c,fc
    return SolveResult((a+b)/2,f((a+b)/2),'iteration_limit','bisection',max_iter)

def newton(f,df,x0,tol=1e-10,max_iter=100):
    x=float(x0)
    for i in range(max_iter):
        fx=f(x); d=df(x)
        if abs(fx)<=tol:return SolveResult(x,fx,'converged','newton',i,residual=abs(fx))
        if abs(d)<1e-15: raise ZeroDivisionError('derivative too small')
        xn=x-fx/d
        if abs(xn-x)<=tol:return SolveResult(xn,f(xn),'converged','newton',i+1,residual=abs(f(xn)))
        x=xn
    return SolveResult(x,f(x),'iteration_limit','newton',max_iter)

def secant(f,x0,x1,tol=1e-10,max_iter=100):
    a=float(x0);b=float(x1);fa=f(a);fb=f(b)
    for i in range(max_iter):
        if abs(fb)<=tol:return SolveResult(b,fb,'converged','secant',i,residual=abs(fb))
        d=fb-fa
        if abs(d)<1e-15: raise ZeroDivisionError('secant denominator too small')
        c=b-fb*(b-a)/d;a,fa,b,fb=b,fb,c,f(c)
    return SolveResult(b,fb,'iteration_limit','secant',max_iter)

def solve_nonlinear_system(F,J,x0,tol=1e-9,max_iter=50):
    from .linear import solve_linear
    x=list(map(float,x0))
    for it in range(max_iter):
        f=list(F(x)); norm=math.sqrt(sum(v*v for v in f))
        if norm<tol:return SolveResult(x,norm,'converged','newton_system',it,residual=norm)
        dx=solve_linear(J(x),[-v for v in f]).solution;x=[x[i]+dx[i] for i in range(len(x))]
    norm=math.sqrt(sum(v*v for v in F(x)));return SolveResult(x,norm,'iteration_limit','newton_system',max_iter,residual=norm)
