from .core import SolveResult
from itertools import combinations

def solve_knapsack(values,weights,capacity):
    n=len(values)
    if len(weights)!=n or capacity<0 or any(int(w)!=w or w<0 for w in weights):raise ValueError('weights must be nonnegative integers')
    cap=int(capacity);dp=[(0,[]) for _ in range(cap+1)]
    for i,(v,w) in enumerate(zip(values,weights)):
        for c in range(cap,int(w)-1,-1):
            cand=dp[c-int(w)][0]+v
            if cand>dp[c][0]:dp[c]=(cand,dp[c-int(w)][1]+[i])
    val,items=max(dp,key=lambda z:z[0]);x=[int(i in items) for i in range(n)]
    return SolveResult(x,float(val),'optimal','0_1_knapsack_dynamic_programming',metadata={'selected_indices':items,'weight':sum(weights[i] for i in items)})

def solve_tsp(distance,start=0):
    n=len(distance)
    if n<2 or any(len(r)!=n for r in distance):raise ValueError('distance matrix must be square and n>=2')
    # Held-Karp exact DP O(n^2 2^n).
    dp={(1<<start,start):(0.0,[start])}
    for size in range(1,n):
        for (mask,last),(cost,path) in list(dp.items()):
            if mask.bit_count()!=size:continue
            for nxt in range(n):
                if mask>>nxt&1:continue
                key=(mask|1<<nxt,nxt);cand=cost+distance[last][nxt]
                if key not in dp or cand<dp[key][0]:dp[key]=(cand,path+[nxt])
    full=(1<<n)-1;best=min((dp[(full,j)][0]+distance[j][start],dp[(full,j)][1]+[start]) for j in range(n) if j!=start)
    return SolveResult(best[1],float(best[0]),'optimal','held_karp_tsp',metadata={'complexity':'O(n^2 2^n)','cities':n})

def maximum_independent_set(n,edges):
    adj=[0]*n
    for u,v in edges:adj[u]|=1<<v;adj[v]|=1<<u
    best=0
    for mask in range(1<<n):
        if mask.bit_count()<=best.bit_count():continue
        ok=True
        for i in range(n):
            if mask>>i&1 and (adj[i]&mask):ok=False;break
        if ok:best=mask
    x=[(best>>i)&1 for i in range(n)]
    return SolveResult(x,sum(x),'optimal','exhaustive_independent_set',metadata={'complexity':'O(2^n)'})

def max_cut(n,edges,weights=None):
    weights=list(weights) if weights is not None else [1.0]*len(edges)
    if len(weights)!=len(edges):raise ValueError('weights mismatch')
    best=-float('inf');bx=None
    # Fix vertex 0 to 0 to remove complement symmetry.
    for mask in range(1<<(n-1)):
        x=[0]+[(mask>>i)&1 for i in range(n-1)];v=sum(w for (a,b),w in zip(edges,weights) if x[a]!=x[b])
        if v>best:best,bx=v,x
    return SolveResult(bx,best,'optimal','exhaustive_max_cut',metadata={'complexity':'O(2^(n-1))'})
