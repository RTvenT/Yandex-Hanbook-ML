import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin


class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        self.param = X.assign(target=y).groupby(by=['city', 'modified_rubrics'])['target'].median().to_dict()
        self.is_fitted_ = True
        
        return self
        
    def predict(self, X):
        prediction = pd.Series(
            zip(X['city'], X['modified_rubrics'])
        ).map(self.param)
        
        return prediction
        