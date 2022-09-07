from sklearn.ensemble import (
        RandomForestRegressor, 
        AdaBoostRegressor, 
        ExtraTreesRegressor, 
        GradientBoostingRegressor
)
from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        BayesianRidge,
        Lasso,
        ElasticNet,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor

_all_regressors = {
        'lir': {
            'regress__estimator': LinearRegression(),
        },
        'lightgbm': {
            'regress__estimator': LGBMRegressor(),
            'regress__estimator__random_state': 64
        },
        'gbr': {
            'regress__estimator': GradientBoostingRegressor(),
            'regress__estimator__random_state': 64
        },
        'rf': {
            'regress__estimator': RandomForestRegressor(),
            'regress__estimator__random_state': 64
        },
        'et': {
            'regress__estimator': ExtraTreesRegressor(),
            'regress__estimator__random_state': 64
        },
        'ada': {
            'regress__estimator': AdaBoostRegressor(),
            'regress__estimator__random_state': 64
        },
        'knn': {
            'regress__estimator': KNeighborsRegressor(),
        },
        'ridge': {
            'regress__estimator': Ridge(),
            'regress__estimator__random_state': 64
        },
        'br': {
            'regress__estimator': BayesianRidge(),
        },
        'lasso': {
            'regress__estimator': Lasso(),
            'regress__estimator__random_state': 64
        },
        'en': {
            'regress__estimator': ElasticNet(),
            'regress__estimator__random_state': 64
        },
        'dummy': {
            'regress__estimator': DummyRegressor(),
        },
}





