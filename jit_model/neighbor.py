# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import ElasticNet
from scipy.spatial.distance import mahalanobis
from .weight import weight_func

class BaseNeighborSearch():
    def __init__(self,ks_ratio=[0.05,0.2,0.3,0.4,0.5]):
        self.ks_ratio = ks_ratio

    def search(query,X):
        raise NotImplementedError("search method must be overrided.")

class SparseSampleSearch(BaseNeighborSearch):
    """Sparse Sample Search for SSR-JIT modeling
    """
    def __init__(self,alpha=0.3,l1_ratio=0.1,max_iter=2000):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

    def search(self,query,X):
        # fit ElasticNet
        lm = ElasticNet(alpha=self.alpha,l1_ratio=self.l1_ratio,max_iter=self.max_iter)
        lm.fit(X.T,query)
        # extract information
        local_indices = np.argwhere(lm.coef_ != 0).T[0]
        weight = np.abs(lm.coef_)[local_indices]
        weight /= weight.sum()  #normalize
        X_local = X[local_indices,:]
        return X_local,weight,local_indices

class EuclideanSearch(BaseNeighborSearch):
    def __init__(self):
        raise NotImplementedError()

    def search(self,query,X):
        pass

class MahalanobisSearch(BaseNeighborSearch):
    def __init__(self):
        super().__init__()

    def search(self,query,X):
        # precision matrix
        vi = np.linalg.inv(np.cov(X.T))
        dist = []
        # measure distance
        for idx,x in enumerate(X):
            mdist = mahalanobis(query,x,vi)
            dict_ = dict(index=idx,distance=mdist)
            dist.append(dict_)

        n_samples = X.shape[0]
        ks = [int(n_samples*kr) for kr in self.ks_ratio]

        X_local_list = []
        weight_list = []
        local_indices_list = []
        for k in ks:
            if k == 0:
                raise ValueError("num of train data is too low.")
            # sort & limit
            dist_s = sorted(dist,key=lambda x: x["distance"])[:k]
            local_indices = [ds["index"] for ds in dist_s]
            X_local = X[local_indices,:]
            local_distance = [ds["distance"] for ds in dist_s]
            weight = weight_func(query,X_local,local_distance)
            X_local_list.append(X_local)
            weight_list.append(weight)
            local_indices_list.append(local_indices)

        return X_local_list,weight_list,local_indices_list
