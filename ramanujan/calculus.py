import math
from .core import SolveResult

def derivative(f,x,h=1e-5):
    if h<=0: raise ValueError('h must be positive')
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)

def integrate_simpson(f,a,b,n=1000):
    if n<=0: raise ValueError('n must be positive')
    if n%2:n+=1
    h=(b-a)/n;s=f(a)+f(b)+4*sum(f(a+i*h) for i in range(1,n,2))+2*sum(f(a+i*h) for i in range(2,n,2))
    return s*h/3

def adaptive_simpson(f,a,b,tol=1e-9,max_depth=20):
    def simp(a,b,fa,fm,fb):return (b-a)*(fa+4*fm+fb)/6
    fa=f(a);fb=f(b);m=(a+b)/2;fm=f(m);whole=simp(a,b,fa,fm,fb)
    def rec(a,b,fa,fm,fb,S,t,d):
        m=(a+b)/2;l=(a+m)/2;r=(m+b)/2;fl=f(l);fr=f(r);L=simp(a,m,fa,fl,fm);R=simp(m,b,fm,fr,fb)
        if d<=0 or abs(L+R-S)<=15*t:return L+R+(L+R-S)/15
        return rec(a,m,fa,fl,fm,L,t/2,d-1)+rec(m,b,fm,fr,fb,R,t/2,d-1)
    return rec(a,b,fa,fm,fb,whole,tol,max_depth)

def rk4(f,t0,y0,t1,steps=1000):
    if steps<=0:raise ValueError('steps must be positive')
    h=(t1-t0)/steps;t=t0;y=list(y0) if isinstance(y0,(list,tuple)) else [float(y0)];out=[(t,list(y))]
    def add(a,b,s):return [a[i]+s*b[i] for i in range(len(a))]
    for _ in range(steps):
        k1=list(f(t,y));k2=list(f(t+h/2,add(y,k1,h/2)));k3=list(f(t+h/2,add(y,k2,h/2)));k4=list(f(t+h,add(y,k3,h)))
        y=[y[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(len(y))];t+=h;out.append((t,list(y)))
    return SolveResult(out,None,'integrated','runge_kutta_4',steps)
