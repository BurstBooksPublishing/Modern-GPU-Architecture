from collections import deque
def avg_shortest_path(adj):
    N = len(adj)
    total = 0
    for s in range(N):
        dist = [-1]*N
        q = deque([s]); dist[s]=0
        while q:
            u = q.popleft()
            for v,cap in enumerate(adj[u]):
                if cap and dist[v]==-1:
                    dist[v]=dist[u]+1; q.append(v)
        total += sum(d for d in dist if d>0)
    return total/(N*(N-1))  # average hops per ordered pair

def bisection_bandwidth(adj, Bw):
    N = len(adj); half = N//2
    cut = 0
    S = set(range(half))
    for i in S:
        for j in range(N):
            if j not in S:
                cut += adj[i][j]  # capacity count
    return cut * Bw  # total capacity across cut

# Example usage: adj as 0/1 capacity matrix; Bw in GB/s
# # adj = [[0,1,1,...], ...]; print(avg_shortest_path(adj), bisection_bandwidth(adj, 25.0))