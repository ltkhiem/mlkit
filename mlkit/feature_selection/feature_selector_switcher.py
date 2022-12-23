import sklearn
from packaging import version
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelectorSwitcher(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None, **kwargs): 
        self.model.fit(X,y)
        return self

    def predict(self, X, y=None):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def score(self, X, y, **fit_params):
        return self.model.score(X, y, **fit_params)

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def get_feature_names_out(self, input_features=None):
        if version.parse(sklearn.__version__) < version.parse('1.0'):
            return input_features[self.model.get_support()]
        return self.model.get_feature_names_out(input_features)

    def get_support(self, indices=False):
        return self.model.get_support(indices)

    def inverse_transform(X):
        return self.model.inverse_transform(X)

