# -*- coding: utf-8 -*-
import numpy as np

class DataBase():
    def __init__(self,X,y):
        self.X = X
        self.y = y

    def delete():
        pass

    def add(self,x_new,y_new):
        self.X = np.vstack((self.X,x_new))
        self.y = np.hstack((self.y,y_new))

    def update(self,x_new,y_new,y_hat):
        """
        TODO: DMIの実装
        """
        self.add(x_new,y_new)
