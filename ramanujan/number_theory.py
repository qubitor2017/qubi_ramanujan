import math

def gcd(a,b):
    while b:a,b=b,a%b
    return abs(a)
def extended_gcd(a,b):
    if b==0:return (abs(a),1 if a>=0 else -1,0)
    g,x1,y1=extended_gcd(b,a%b);return g,y1,x1-(a//b)*y1
def mod_inverse(a,m):
    g,x,_=extended_gcd(a,m)
    if g!=1:raise ValueError('inverse does not exist')
    return x%m
def is_prime(n):
    if n<2:return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n%p==0:return n==p
    d=n-1;s=0
    while d%2==0:s+=1;d//=2
    # Deterministic Miller-Rabin for unsigned 64-bit integers.
    bases=(2,325,9375,28178,450775,9780504,1795265022) if n<2**64 else (2,3,5,7,11,13,17)
    for a in bases:
        if a%n==0:continue
        x=pow(a,d,n)
        if x in (1,n-1):continue
        for _ in range(s-1):
            x=x*x%n
            if x==n-1:break
        else:return False
    return True
def prime_factors(n):
    if n==0:raise ValueError('zero has no finite prime factorization')
    n=abs(n);out=[];d=2
    while d*d<=n:
        while n%d==0:out.append(d);n//=d
        d=3 if d==2 else d+2
    if n>1:out.append(n)
    return out
def chinese_remainder(remainders,moduli):
    if len(remainders)!=len(moduli) or not moduli:raise ValueError('invalid CRT inputs')
    x=0;M=1
    for a,m in zip(remainders,moduli):
        if m<=0:raise ValueError('moduli must be positive')
        g=gcd(M,m)
        if (a-x)%g:raise ValueError('inconsistent congruences')
        l=M//g*m;k=((a-x)//g*mod_inverse(M//g,m//g))%(m//g);x=(x+M*k)%l;M=l
    return x,M
def integer_sqrt(n):
    if n<0:raise ValueError('n must be nonnegative')
    return math.isqrt(n)
