# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LassoLars,LassoCV

import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError
from .r import get_robject

class BaseFeatureSelect():
    def __init__(self):
        pass

    def select(self,X,y,weight):
        raise NotImplementedError()

class RWeightedLassoSelect(BaseFeatureSelect):
    """
    R glmnetによる実装 CVなし
    """
    def __init__(self,glmnet,stats,n_features=2,alpha=0.3):
        self.glmnet = glmnet
        self.stats = stats
        self.n_features = n_features

        # alpha: 0(ridge) < 1(lasso)
        self.alpha = alpha

        # array conversion activation
        numpy2ri.activate()
        pandas2ri.activate()

    def select(self,X,y,weight):
        # import pdb; pdb.set_trace()
        res = self.glmnet.glmnet(X,y[np.newaxis,:],weights=weight,alpha=self.alpha)
        coef_array = np.array(get_robject(res,"beta"))
        import pdb; pdb.set_trace()
        path_idx = np.argwhere((coef_array != 0).sum(axis=0) <= self.n_features)[-1,0]
        coef = coef_array[:,path_idx]
        f_indices = np.argwhere(coef != 0).T[0]
        if len(f_indices) == 0:
            f_indices = self.select(X,y,alpha=alpha * 0.01)
        return f_indices

class RWeightedLassoSelectCV(BaseFeatureSelect):
    """
    R glmnetによる実装 CVあり
    """
    def __init__(self,alpha=0.3,nfold=5,min_lamb="lambda.min"):
        self.glmnet = importr("glmnet")
        self.stats = importr("stats")
        self.base = importr("base")

        # alpha: 0(ridge) < 1(lasso)
        self.alpha = alpha
        self.nfold = nfold
        self.min_lamb = min_lamb

        # array conversion activation
        numpy2ri.activate()
        pandas2ri.activate()

    def select(self,X,y,weight):
        res = self.glmnet.cv_glmnet(X,y[np.newaxis,:],weights=weight,nfold=self.nfold,alpha=self.alpha)
        coef = np.array(self.base.as_matrix(self.stats.coef(res,s=get_robject(res,self.min_lamb)))).T[0][1:]
        f_indices = np.argwhere(coef != 0).T[0]
        if len(f_indices) == 0:
            error = np.inf

        # extract error
        cvm = np.array(get_robject(res,"cvm"))
        lambda_ = np.array(get_robject(res,"lambda"))
        lambda_1se = np.array(get_robject(res,self.min_lamb))[0]
        error = cvm[np.where(lambda_ == lambda_1se)[0][0]]/y.size

        return f_indices,error


class LarsSelect(BaseFeatureSelect):
    """
    180220: weight考慮できない
    """
    def __init__(self,n_features=2):
        self.n_features = n_features

    def select(self,X,y,weight,alpha=0.01):
        lars = LassoLars(normalize=False,alpha=alpha)
        lars.fit(X,y)
        path_idx = np.argwhere((lars.coef_path_ != 0).sum(axis=0) <= self.n_features)[-1,0]
        coef = lars.coef_path_[:,path_idx]
        f_indices = np.argwhere(coef != 0).T[0]
        if len(f_indices) == 0:
            f_indices = self.select(X,y,alpha=alpha * 0.01)
        return f_indices

class LassoCvSelect(BaseFeatureSelect):
    """
    180220: weight考慮できない
    """
    def __init__(self,cv=2):
        self.cv = cv

    def select(self,X,y,weight):
        lm = LassoCV(cv=self.cv,normalize=False,max_iter=2000)
        lm.fit(X,y)
        f_indices = np.argwhere(lm.coef_ != 0).T[0]
        return f_indices
