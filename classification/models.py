from sklearn.ensemble import (
        RandomForestClassifier, 
        AdaBoostClassifier, 
        ExtraTreesClassifier, 
        GradientBoostingClassifier
)
from sklearn.linear_model import (
        LogisticRegression,
        RidgeClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis
)
from lightgbm import LGBMClassifier

_all_clfs = {
        'lr': {
            'classify__estimator': LogisticRegression(),
            'classify__estimator__random_state': 64
        },
        'lightgbm': {
            'classify__estimator': LGBMClassifier(),
            'classify__estimator__random_state': 64
        },
        'gbc': {
            'classify__estimator': GradientBoostingClassifier(),
            'classify__estimator__random_state': 64
        },
        'rf': {
            'classify__estimator': RandomForestClassifier(),
            'classify__estimator__random_state': 64
        },
        'et': {
            'classify__estimator': ExtraTreesClassifier(),
            'classify__estimator__random_state': 64
        },
        'ada': {
            'classify__estimator': AdaBoostClassifier(),
            'classify__estimator__random_state': 64
        },
        'nb': {
            'classify__estimator': GaussianNB(),
        },
        'knn': {
            'classify__estimator': KNeighborsClassifier(),
        },
        'lda': {
            'classify__estimator': LinearDiscriminantAnalysis(),
        },
        'qda': {
            'classify__estimator': QuadraticDiscriminantAnalysis(),
        },
        'ridge': {
            'classify__estimator': RidgeClassifier(),
            'classify__estimator__random_state': 64
        },
        'dt': {
            'classify__estimator': DecisionTreeClassifier(),
            'classify__estimator__random_state': 64
        },
        'svm': {
            'classify__estimator': SVC(),
            'classify__estimator__random_state': 64
        },
        'dummy': {
            'classify__estimator': DummyClassifier(),
            'classify__estimator__random_state': 64
        },
}


