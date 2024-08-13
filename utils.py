
import matplotlib.pyplot as plt
import gudhi
from gudhi.wasserstein import wasserstein_distance
import torch
from torch.optim.lr_scheduler import LambdaLR

def myloss(pts):
    rips = gudhi.RipsComplex(points=pts, max_edge_length=0.5)
    # .5 because it is faster and, experimentally, the cycles remain smaller
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    i = st.flag_persistence_generators()
    if len(i[1]) > 0:
        i1 = torch.tensor(i[1][0])  # pytorch sometimes interprets it as a tuple otherwise
    else:
        i1 = torch.empty((0, 4), dtype=int)
    # Same as the finite part of st.persistence_intervals_in_dimension(1), but differentiable
    diag1 = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)
    # Total persistence is a special case of Wasserstein distance
    perstot1 = wasserstein_distance(diag1, [], order=1, enable_autodiff=True)
    # Stay within the unit disk
    disk = (pts ** 2).sum(-1) - 1
    disk = torch.max(disk, torch.zeros_like(disk)).sum()
    return -perstot1 + 1 * disk