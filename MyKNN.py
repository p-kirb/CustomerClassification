
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from HelperFunctions import scaleNumericalFeatures
import settings
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels



class MyKNN(BaseEstimator, ClassifierMixin):

    #gaussian model's parameters are fixed as the defaults.
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        
    def fit(self, X, Y):

        #X, Y = check_X_y(X, Y)

        self.classes_ = unique_labels(Y)

        self.X_, self.scaler = scaleNumericalFeatures(X)
        self.Y_ = Y

        self.model.fit(self.X_[settings.categoricalFeatures], self.Y_)

        return self
    
    def predict_proba(self, X):

        scaled, _ = scaleNumericalFeatures(X, self.scaler)

        probs = self.model.predict_proba(scaled)

        return probs


    def predict(self, X):

        check_is_fitted(self)

        X = check_array(X)

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)     #returns index of largest probability (as labels are 0 and 1, indexes match labels)

