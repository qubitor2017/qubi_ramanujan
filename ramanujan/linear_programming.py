"""Small LP/MILP reference solvers. Vertex enumeration is exponential."""
from itertools import combinations
from .core import SolveResult
from .linear import solve_linear

def solve_lp_vertex(c,A_ub=None,b_ub=None,A_eq=None,b_eq=None,bounds=None,tol=1e-8):
    """Minimize c.x subject to <=, = and box bounds; intended for small n only."""
    n=len(c);A=[];b=[]
    for row,rhs in zip(A_ub or [],b_ub or []):A.append(list(row));b.append(float(rhs))
    for i,(lo,hi) in enumerate(bounds or [(0,None)]*n):
        if hi is not None:
            row=[0.0]*n;row[i]=1;A.append(row);b.append(float(hi))
        if lo is not None:
            row=[0.0]*n;row[i]=-1;A.append(row);b.append(-float(lo))
    eq=list(A_eq or []);be=list(b_eq or [])
    # Equality-constrained vertex enumeration: select n-rank(eq) active inequalities.
    need=n-len(eq)
    if need<0:raise ValueError('too many equality constraints')
    best=None;bestv=float('inf');checked=0
    for ids in combinations(range(len(A)),need):
        M=eq+[A[i] for i in ids];rhs=be+[b[i] for i in ids]
        if len(M)!=n:continue
        try:x=solve_linear(M,rhs).solution
        except ValueError:continue
        checked+=1
        if any(sum(row[j]*x[j] for j in range(n))>rhs0+tol for row,rhs0 in zip(A,b)):continue
        if any(abs(sum(row[j]*x[j] for j in range(n))-rhs0)>tol for row,rhs0 in zip(eq,be)):continue
        val=sum(c[i]*x[i] for i in range(n))
        if val<bestv:best,bestv=x,val
    if best is None:return SolveResult(None,None,'infeasible_or_unbounded','vertex_enumeration',checked)
    return SolveResult(best,bestv,'optimal_for_enumerated_vertices','vertex_enumeration',checked,metadata={'warning':'small problems only; numerical tolerances apply'})

def branch_and_bound_milp(c,A_ub,b_ub,integrality=None,bounds=None,node_limit=10000,tol=1e-7):
    """Basic branch-and-bound over the LP vertex solver; integer variables default all."""
    n=len(c);ints=set(range(n) if integrality is None else [i for i,v in enumerate(integrality) if v]);base=list(bounds or [(0,None)]*n);stack=[base];best=None;bestv=float('inf');nodes=0
    while stack and nodes<node_limit:
        bd=stack.pop();nodes+=1
        r=solve_lp_vertex(c,A_ub,b_ub,bounds=bd,tol=tol)
        if r.solution is None or r.objective>=bestv-tol:continue
        frac=next((i for i in sorted(ints) if abs(r.solution[i]-round(r.solution[i]))>tol),None)
        if frac is None:best,bestv=r.solution,r.objective;continue
        v=r.solution[frac];lo,hi=bd[frac]
        left=bd[:];left[frac]=(lo, min(hi, int(v//1)) if hi is not None else int(v//1))
        right=bd[:];right[frac]=(max(lo, int(v//1)+1) if lo is not None else int(v//1)+1,hi)
        if left[frac][1] is None or left[frac][0] is None or left[frac][0]<=left[frac][1]:stack.append(left)
        if right[frac][1] is None or right[frac][0] is None or right[frac][0]<=right[frac][1]:stack.append(right)
    return SolveResult(best,bestv if best is not None else None,'optimal' if not stack and best is not None else ('infeasible' if best is None and not stack else 'node_limit'),'branch_and_bound',nodes,metadata={'nodes_remaining':len(stack),'integer_variables':sorted(ints)})
