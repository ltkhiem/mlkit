from sklearn.feature_selection import(
    RFECV,
    SequentialFeatureSelector
)

_all_selectors = {
    'rfe_reg': {
        'feature_select__model': RFECV(estimator=None),
        'feature_select__model__n_jobs': -1,
        'feature_select__model__step': 1,
        'feature_select__model__scoring': 'r2',
        'feature_select__model__min_features_to_select': 1
    },
    'rfe_clf': {
        'feature_select__model': RFECV(estimator=None),
        'feature_select__model__n_jobs': -1,
        'feature_select__model__step': 1,
        'feature_select__model__scoring': 'accuracy',
        'feature_select__model__min_features_to_select': 1
    },
    'sfsf_reg': {
        'feature_select__model': SequentialFeatureSelector(estimator=None),
        'feature_select__model__n_jobs': -1,
        'feature_select__model__step': 1,
        'feature_select__model__scoring': 'r2',
        'feature_select__model__n_features_to_select': "auto",
        'feature_select__model__tol': 0.01,
        'feature_select__model__direction': "forward",

    },
    'sfsf_clf': {
        'feature_select__model': SequentialFeatureSelector(estimator=None),
        'feature_select__model__n_jobs': -1,
        'feature_select__model__step': 1,
        'feature_select__model__scoring': 'accuracy',
        'feature_select__model__n_features_to_select': "auto",
        'feature_select__model__tol': 0.01,
        'feature_select__model__direction': "forward",

    },
    'sfsb': {

    }

}
