from collections import deque
from .core import SolveResult
import heapq, math

def dijkstra(graph,source):
    dist={v:math.inf for v in graph};prev={v:None for v in graph};dist[source]=0;q=[(0,source)]
    while q:
        d,u=heapq.heappop(q)
        if d!=dist[u]:continue
        for v,w in graph[u].items():
            if w<0:raise ValueError('Dijkstra requires nonnegative weights')
            nd=d+w
            if nd<dist.get(v,math.inf):dist[v]=nd;prev[v]=u;heapq.heappush(q,(nd,v))
    return SolveResult({'distance':dist,'predecessor':prev},None,'solved','dijkstra')

def bellman_ford(edges,vertices,source):
    d={v:math.inf for v in vertices};p={v:None for v in vertices};d[source]=0
    for _ in range(len(vertices)-1):
        changed=False
        for u,v,w in edges:
            if d[u]+w<d[v]:d[v]=d[u]+w;p[v]=u;changed=True
        if not changed:break
    neg=any(d[u]+w<d[v] for u,v,w in edges)
    return SolveResult({'distance':d,'predecessor':p},None,'negative_cycle' if neg else 'solved','bellman_ford')

def floyd_warshall(matrix):
    d=[list(map(float,r)) for r in matrix];n=len(d)
    if any(len(r)!=n for r in d):raise ValueError('matrix must be square')
    for k in range(n):
        for i in range(n):
            for j in range(n):d[i][j]=min(d[i][j],d[i][k]+d[k][j])
    return SolveResult(d,None,'solved','floyd_warshall')

def kruskal_mst(vertices,edges):
    parent={v:v for v in vertices};rank={v:0 for v in vertices}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    chosen=[];cost=0
    for u,v,w in sorted(edges,key=lambda e:e[2]):
        a,b=find(u),find(v)
        if a!=b:
            if rank[a]<rank[b]:a,b=b,a
            parent[b]=a
            if rank[a]==rank[b]:rank[a]+=1
            chosen.append((u,v,w));cost+=w
    status='spanning_tree' if len(chosen)==max(0,len(vertices)-1) else 'disconnected_forest'
    return SolveResult(chosen,cost,status,'kruskal_mst')

def max_flow_edmonds_karp(capacity,source,sink):
    residual={u:dict(vs) for u,vs in capacity.items()}
    for u,vs in list(residual.items()):
        for v,c in vs.items():residual.setdefault(v,{}).setdefault(u,0)
    flow=0
    while True:
        parent={source:None};q=deque([source])
        while q and sink not in parent:
            u=q.popleft()
            for v,c in residual[u].items():
                if c>0 and v not in parent:parent[v]=u;q.append(v)
        if sink not in parent:break
        aug=float('inf');v=sink
        while parent[v] is not None:aug=min(aug,residual[parent[v]][v]);v=parent[v]
        v=sink
        while parent[v] is not None:u=parent[v];residual[u][v]-=aug;residual[v][u]+=aug;v=u
        flow+=aug
    return SolveResult({'value':flow,'residual':residual},flow,'optimal','edmonds_karp')

def bipartite_matching(left,right,adj):
    match={};
    def dfs(u,seen):
        for v in adj.get(u,[]):
            if v in seen:continue
            seen.add(v)
            if v not in match or dfs(match[v],seen):match[v]=u;return True
        return False
    count=0
    for u in left:count+=bool(dfs(u,set()))
    pairs=[(u,v) for v,u in match.items()]
    return SolveResult(pairs,count,'optimal','augmenting_path_matching')
