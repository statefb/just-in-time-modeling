# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import ElasticNet

class BaseNeighborSearch():
    def __init__(self):
        pass

    def search(query,X):
        raise NotImplementedError("search method must be overrided.")

class SparseSampleSearch(BaseNeighborSearch):
    """Sparse Sample Search for SSR-JIT modeling
    """
    def __init__(self,alpha=0.8,l1_ratio=0.1,iter=2000):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.iter = iter

    def search(query,X):
        # fit ElasticNet
        lm = ElasticNet(alpha=self.alpha,l1_ratio=self.l1_ratio,iter=self.iter)
        lm.fit(X.T,query)
        # extract information
        local_indices = np.argwhere(lm.coef_ != 0)[0]
        weight = lm.coef_[local_indices]
        weight /= weight.sum()  #normalize
        X_local = X[local_indices,:]
        return X_local,weight,local_indices

class EuclideanSearch(BaseNeighborSearch):
    def __init__(self):
        raise NotImplementedError()

    def search(query,X):
        pass

class MahalanobisSearch(BaseNeighborSearch):
    def __init__(self):
        raise NotImplementedError()

    def search(query,X):
        pass
