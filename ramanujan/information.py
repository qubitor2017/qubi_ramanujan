import math

def _norm(p):
    if any(x<0 for x in p) or sum(p)<=0:raise ValueError('probabilities must be nonnegative with positive sum')
    s=sum(p);return [x/s for x in p]
def entropy(p,base=2):
    p=_norm(p);return -sum(x*math.log(x,base) for x in p if x>0)
def cross_entropy(p,q,base=2):
    p=_norm(p);q=_norm(q)
    if len(p)!=len(q):raise ValueError('dimension mismatch')
    if any(qi==0 and pi>0 for pi,qi in zip(p,q)):return math.inf
    return -sum(pi*math.log(qi,base) for pi,qi in zip(p,q) if pi>0)
def kl_divergence(p,q,base=2):return cross_entropy(p,q,base)-entropy(p,base)
def mutual_information(joint):
    rows=len(joint);cols=len(joint[0])
    if any(len(r)!=cols for r in joint):raise ValueError('joint distribution must be rectangular')
    flat=_norm([x for r in joint for x in r]);P=[flat[i*cols:(i+1)*cols] for i in range(rows)];px=[sum(r) for r in P];py=[sum(P[i][j] for i in range(rows)) for j in range(cols)]
    return sum(P[i][j]*math.log(P[i][j]/(px[i]*py[j]),2) for i in range(rows) for j in range(cols) if P[i][j]>0)
