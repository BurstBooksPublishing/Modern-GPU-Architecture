# blocks: dict name -> (x,y,w,h); nets: list of lists of pin coords
def hpwl(blocks, nets):
    total=0.0
    for net in nets:
        xs=[]; ys=[]
        for (bname, px, py) in net: xs.append(blocks[bname][0]+px); ys.append(blocks[bname][1]+py)
        total += (max(xs)-min(xs)) + (max(ys)-min(ys))
    return total

def swap_and_cost(blocks, nets, a, b):
    # swap centers of blocks a and b
    ba=blocks[a]; bb=blocks[b]
    blocks[a], blocks[b] = (bb[0],ba[1],ba[2],ba[3]), (ba[0],bb[1],bb[2],bb[3])
    cost = hpwl(blocks, nets)
    # undo swap for caller to decide
    blocks[a], blocks[b] = ba, bb
    return cost

# usage: simulated annealing loop would call swap_and_cost to accept/reject moves