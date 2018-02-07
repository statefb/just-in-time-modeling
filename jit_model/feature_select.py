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

    def select(self,X,y):
        lars = LassoLars(normalize=True)
        lars.fit(X,y)
        lars.coef_path_
        return f_indices

class LassoCvSelect(BaseFeatureSelect):
    def __init__(self,cv=2):
        self.cv = cv

    def select(self,X,y):
        lm = LassoCV(cv=self.cv)
        lm.fit(X,y)
        
