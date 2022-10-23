import pandas as pd
import numpy as np


# the Brier loss function:
def brier_loss(y, p, k):
    loss = 0
    for i in range(k):
        if y == i:
            loss += (1 - p[i]) ** 2
        else:
            loss += p[i] ** 2
    return loss


# Arithmetic average of numbers given on the log10 scale:
def log_mean(x):
    m = np.max(x)
    return m + np.log10(np.mean(np.exp(np.log(10) * (x - m))))