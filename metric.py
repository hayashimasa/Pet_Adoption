import numpy as np
# import torch

def quadratic_weighted_kappa(pred, target):
    N = 5
    O = np.zeros((N, N))
    w = np.array([[(i-j)**2 / (N-1)**2 for j in range(N)] for i in range(N)])
    try:
        pred, target = pred.round().long(), target.long()
    except:
        pred, target = np.round(pred).astype(int), target.astype(int)
    for i, y in enumerate(target):
        O[y][min(pred[i], 4)] += 1
    E = np.sum(O, axis=1).reshape(N,1) @ np.sum(O, axis=0).reshape(1,N)
    E *= np.sum(O)/ np.sum(E)
    kappa = 1 - np.sum(w * O) / np.sum(w * E)
    return kappa
