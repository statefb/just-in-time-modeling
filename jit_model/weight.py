# -*- coding: utf-8 -*-
import numpy as np

def weight_func(query,X_local,dist):
    # h = determine_bandwidth(query,X_local)
    x_tri = dist/max(dist)
    weight = tricube(x_tri)
    return weight

def tricube(x):
    weight = np.zeros(x.shape[0])
    for i,xx in enumerate(x):
        weight[i] = (1 - np.linalg.norm(xx)**3)**3
    weight[weight < 0] = 0
    return weight

def determine_bandwidth(query,X_local):
    half_width = np.linalg.norm(X_local[-1] - query)
    return half_width * 2
