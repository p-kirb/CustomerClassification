
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import settings
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels



class MixedNB(BaseEstimator, ClassifierMixin):
    alpha = 1.0
    #gaussian model's parameters are fixed as the defaults.
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.gaussianModel = GaussianNB()
        self.categoricalModel = CategoricalNB(alpha=self.alpha, force_alpha=True)
    def fit(self, X, Y):

        #X, Y = check_X_y(X, Y)

        self.classes_ = unique_labels(Y)

        self.X_ = X
        self.Y_ = Y

        self.categoricalModel.fit(X[settings.categoricalFeatures], Y)
        self.gaussianModel.fit(X[settings.numericalFeatures], Y)

        return self
    
    def predict_proba(self, X):

        catProb = self.categoricalModel.predict_proba(X[settings.categoricalFeatures])
        numProb = self.gaussianModel.predict_proba(X[settings.numericalFeatures])

        overallPreNorm = (catProb * numProb) / self.gaussianModel.class_prior_
        totSum = np.sum(overallPreNorm, 1)

        overall = overallPreNorm / totSum[:,None]           #final division step for normalising the probabilites back so their sum is 1

        return overall


    def predict(self, X):

        check_is_fitted(self)

        #X = check_array(X)

        probs = self.predict_proba(X)

        return np.argmax(probs, axis=1)     #returns index of largest probability (as labels are 0 and 1, indexes match labels)


    def set_params(self, **params):
        self.categoricalModel.set_params(**params)
        return self


    #only attribute to change, so always returns alpha
    #def get_params(self):
    #    return {"alpha": self.alpha}