from sklearn.base import BaseEstimator

class RegressorSwitcher(BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs): 
        self.estimator.fit(X,y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
