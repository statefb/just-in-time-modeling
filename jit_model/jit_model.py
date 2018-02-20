# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from .database import DataBase
from .neighbor import *
from .lowess import WeightedLinearRegression
from .feature_select import *

class JitModel(BaseEstimator,RegressorMixin):
    def __init__(self,neighbor_search="sparse_sample",feature_select="weighted_lasso",\
        pre_normalized=True):
        """
        Parameters
        ----------
        neighbor_search : str or BaseNeighborSearch object
        n_features : int
        pre_normalized : bool

        """
        #TODO: 各モデルのパラメータを辞書形式で受け取る
        self.neighbor_search = neighbor_search
        self.pre_normalized = pre_normalized
        self.feature_select = feature_select

    def fit(self,X,y):
        if not self.pre_normalized:
            X,y,scale_param = normalize(X,y)
            self.scale_param_ = scale_param
        self.database_ = DataBase(X,y)
        self.neighbor_search_ = _get_neighbor_search(self.neighbor_search)
        self.feature_select_ = _get_feature_select(self.feature_select)
        self.results_ = []  #for store prediction result
        return self

    def predict(self,X,y=None):
        """
        Parameters
        ----------
        X : 2D array
            input variables
        y : 1D array
            observed output variable

        Returns
        -------
        yhat : 1D array
            predicted output variable

        """
        if not self.pre_normalized:
            X,y = normalize(X,y,self.scale_param_)
        yhat = np.empty(X.shape[0])

        for idx,query in tqdm(enumerate(X)):
            # get part of samples for local regression
            X_local_list,weight_list,local_indices_list = \
                self.neighbor_search_.search(query,self.database_.X)
            error_list = []
            feature_indices_list = []
            for X_local,weight,local_indices in zip(X_local_list,weight_list,local_indices_list):
                y_local = self.database_.y[local_indices]
                # feature selection
                feature_indices,error = self.feature_select_.select(X_local,y_local,weight)

                error_list.append(error)
                feature_indices_list.append(feature_indices)

            # determine feature indices which minimize error
            min_set = sorted(zip(error_list,feature_indices_list,local_indices_list,weight_list))[0]
            min_set = sorted(zip(error_list,feature_indices_list,local_indices_list,weight_list),key=lambda x:x[0])[0]
            error = min_set[0]
            feature_indices = min_set[1]
            local_indices = min_set[2]
            weight = min_set[3]

            X_local = self.database_.X[local_indices,:][:,feature_indices]
            y_local = self.database_.y[local_indices]

            # weighted regression using deterimined local sample and features
            try:
                local_model = LinearRegression().fit(X_local,y_local,sample_weight=weight)
            except:
                import pdb; pdb.set_trace()
                X_local,weight,local_indices = \
                    self.neighbor_search_.search(query,self.database_.X)
            yhat[idx] = local_model.predict(query[np.newaxis,feature_indices])

            # result
            res = dict(
                yhat=yhat[idx],
                feature_selection_error=error,
                feature_indices=feature_indices,
                coef=local_model.coef_,
                X_local=X_local,
                y_local=y_local,
                local_indices=local_indices,
                weight=weight,
            )
            self.results_.append(res)

            # update database if observation given
            if y is not None:
                self.update(query,y[idx],yhat[idx])

        return yhat

    def update(self,x_new,y_new,y_hat):
        """Update Database
        Parameters
        ----------
        x_new
        y_new
        y_hat
        """
        #TODO: 単体更新，一括更新含む
        self.database_.update(x_new,y_new,y_hat)

def _get_neighbor_search(neighbor_search):
    """
    neighbor_search : str or BaseNeighborSearch object
    """
    if isinstance(neighbor_search,BaseNeighborSearch):
        return neighbor_search
    elif type(neighbor_search) != str:
        raise ValueError("neighbor_search must be str or BaseNeighborSearch object")

    if neighbor_search == "sparse_sample":
        return SparseSampleSearch()
    elif neighbor_search == "euclidean":
        return EuclideanSearch()
    elif neighbor_search == "mahalanobis":
        return MahalanobisSearch()
    else:
        raise NotImplementedError("given neighbor_search not supported")

def _get_feature_select(feature_select):
    if isinstance(feature_select,BaseFeatureSelect):
        return feature_select
    elif type(feature_select) != str:
        raise ValueError("feature_select must be str or BaseFeatureSelect object")
    if feature_select == "lars":
        return LarsSelect(n_features=3)
    elif feature_select == "lassocv":
        return LassoCvSelect(cv=5)
    elif feature_select == "weighted_lasso":
        return RWeightedLassoSelectCV()
    else:
        raise NotImplementedError()
