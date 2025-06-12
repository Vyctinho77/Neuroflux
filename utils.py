import numpy as np

def binary_cross_entropy(pred, target):
    pred = np.clip(pred, 1e-7, 1 - 1e-7)
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
