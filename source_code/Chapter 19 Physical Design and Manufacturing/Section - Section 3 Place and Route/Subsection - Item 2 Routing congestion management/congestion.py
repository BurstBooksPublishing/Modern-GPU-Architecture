# compute D and K per bin; nets is list of nets with bbox and criticality
def compute_congestion(bins, nets):
    D = [0]*len(bins); K = [bin.capacity for bin in bins]
    for net in nets:
        bins_hit = net.estimate_bins()               # bbox-based estimation
        for i in bins_hit:
            D[i] += net.track_demand * net.criticality_weight()
    C = [max(0,(D[i]-K[i])/K[i]) for i in range(len(bins))]
    return C

def select_ripup_candidates(C, bins, nets, thresh=0.2):
    # choose nets that contribute most to high-C bins, prefer noncritical nets
    score = {net:0 for net in nets}
    for i,c in enumerate(C):
        if c>thresh:
            for net in bins[i].nets: score[net] += c*net.width
    # return top-k noncritical nets
    return sorted(nets, key=lambda n: (score[n], n.slack), reverse=True)[:50]