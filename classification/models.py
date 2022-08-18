from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_all_clfs = {
        'RandomForestClassifier': {
            'classify__estimator': RandomForestClassifier(),
            'classify__estimator__random_state': 64
        },
        'LogisticRegression': {
            'classify__estimator': LogisticRegression(),
            'classify__estimator__random_state': 64
        }
}


