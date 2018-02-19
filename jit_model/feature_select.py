# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LassoLars,LassoCV

class BaseFeatureSelect():
    def __init__(self):
        pass

    def select(self,X,y):
        raise NotImplementedError()

class LarsSelect(BaseFeatureSelect):
    def __init__(self,n_features=2):
        self.n_features = n_features

    def select(self,X,y,alpha=0.01):
        lars = LassoLars(normalize=False,alpha=alpha)
        lars.fit(X,y)
        path_idx = np.argwhere((lars.coef_path_ != 0).sum(axis=0) <= self.n_features)[-1,0]
        coef = lars.coef_path_[:,path_idx]
        f_indices = np.argwhere(coef != 0).T[0]
        if len(f_indices) == 0:
            f_indices = self.select(X,y,alpha=alpha * 0.01)
        return f_indices

class LassoCvSelect(BaseFeatureSelect):
    def __init__(self,cv=2):
        self.cv = cv

    def select(self,X,y):
        lm = LassoCV(cv=self.cv,normalize=False,max_iter=2000)
        lm.fit(X,y)
        f_indices = np.argwhere(lm.coef_ != 0).T[0]
        return f_indices
