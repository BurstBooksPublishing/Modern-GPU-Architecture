def choose_tile_block(S_tile_bytes, m, n, k, bytes_per_elem=2):
    # compute max block sizes bm,bn,bk that fit: (bm*bk + bk*bn + bm*bn)*elem_size <= S_tile
    # simple greedy: fix bk, maximize bm and bn equally
    for bk in range(min(k,256),0,-1):
        max_pair = (S_tile_bytes/bytes_per_elem)/(bk)  # approx bm+bn+bm*bn/bk term simplified
        bm = bn = int((max_pair)**0.5)
        if bm>=1 and bn>=1:
            return bm, bn, bk
    return 1,1,1  # fallback minimal blocking
# Example: S_tile_bytes = 64*1024  # use \lstinline|S_tile_bytes| in comments above