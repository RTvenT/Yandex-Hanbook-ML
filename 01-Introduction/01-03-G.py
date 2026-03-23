import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        self.msk_mean = y[X['city'] == 'msk'].mean()
        self.spb_mean = y[X['city'] == 'spb'].mean()
        
        self.is_fitted_ = True
        return self

    def predict(self, X=None):
        return (X['city'] == 'msk') * self.msk_mean + (X['city'] == 'spb') * self.spb_mean