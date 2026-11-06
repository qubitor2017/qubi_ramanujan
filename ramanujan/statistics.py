import math

def linear_regression(X,y):
    if not X or len(X)!=len(y):raise ValueError('dimension mismatch')
    n=len(X);p=len(X[0]);A=[[sum(X[k][i]*X[k][j] for k in range(n)) for j in range(p)] for i in range(p)];b=[sum(X[k][i]*y[k] for k in range(n)) for i in range(p)]
    from .linear import solve_linear
    beta=solve_linear(A,b).solution;pred=[sum(row[j]*beta[j] for j in range(p)) for row in X];rss=sum((pred[i]-y[i])**2 for i in range(n))
    return {'coefficients':beta,'predictions':pred,'rss':rss,'method':'normal_equations'}
def covariance_matrix(data):
    if not data:raise ValueError('data required')
    n=len(data);p=len(data[0])
    if n<2 or any(len(r)!=p for r in data):raise ValueError('need rectangular data with at least two rows')
    means=[sum(r[j] for r in data)/n for j in range(p)]
    return [[sum((r[i]-means[i])*(r[j]-means[j]) for r in data)/(n-1) for j in range(p)] for i in range(p)]
def pearson_correlation(x,y):
    if len(x)!=len(y) or len(x)<2:raise ValueError('equal samples of length >=2 required')
    mx=sum(x)/len(x);my=sum(y)/len(y);a=sum((u-mx)*(v-my) for u,v in zip(x,y));b=math.sqrt(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y))
    if b==0:raise ValueError('constant input')
    return a/b
