import mlflow
import mlflow.sklearn 
from tempfile import mkdtemp

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    LeaveOneGroupOut,
)
from sklearn.pipeline import Pipeline

from mlkit.feature_selection import FeatureSelectorSwitcher
from mlkit.classification.models import _all_clfs
from mlkit.feature_selection.selectors import _all_selectors

from ._base import BaseExperiment


class FeatureSelectionExperiment(BaseExperiment):
    def __init__(self,
            experiment_name,
            estimator,
            data,
            target,
            test_data = None,
            use_features = None,
            ignore_features = None,
            mlflow_uri = './',
            experiment_exists_ok = False,
            cv = 'sss',
            stratify_splits = 3,
            stratify_test_size = 0.2,
            group_features = None,
            transformation = False,
            transformation_method = None,
            normalisation = False,
            normalisation_method = None,
            selector = 'rfe',
            run_tags = None,
            random_state = 64,
            use_cache = True
    ):

        super().__init__(experiment_name,
                data,
                target,
                test_data,
                use_features,
                ignore_features,
                mlflow_uri,
                experiment_exists_ok,
                run_tags,
                random_state)
        self._setup_preprocessing(transformation,
                transformation_method,
                normalisation,
                normalisation_method)
        self._setup_cv(cv, 
                group_features,
                stratify_splits,
                stratify_test_size)
        self._setup_estimator(estimator)
        self.selector = selector
        self.use_cache = use_cache
        self._gen_pipeline() 

    def _setup_preprocessing(
            self,
            transformation,
            transformation_method,
            normalisation,
            normalisation_method
    ):
        self.transformation = transformation
        if self.transformation:
            assert transformation_method, "Please specified transformation method"
            self.transformation_method = transformation_method

        self.normalisation = normalisation
        if self.normalisation:
            assert normalisation_method, "Please specified normalisation method"
            self.normalisation_method = normalisation_method

    def _setup_cv(self, 
            cv, 
            group_features,
            stratify_splits,
            stratify_test_size,
    ):
        self.cv = cv
        if self.cv == 'sss':
            self._cv_params = {
                'feature_select__model__cv': StratifiedShuffleSplit(
                                                n_splits = stratify_splits,
                                                test_size = stratify_test_size,
                                                random_state = self.random_state
                                             ),
            }
        elif self.cv == 'logo':
            assert group_features is not None, "'groups' has to be specified to use \
                                             Leave One Group Out split."
            self._cv_params = {
                'feature_select__model__cv': LeaveOneGroupOut(
                                                group_features = group_features
                                             ),
            }

    def _setup_estimator(self, estimator):
        """
        Not considering regressor yet. Classifiers only for now.

        Idea for future: estimator starts with c_ means classifier
                                               r_ means regressor
        """
        self._estimator_params = {}
        _params = _all_clfs[estimator]
        for k, v in _params.items():
            nk = k.replace('classify__', 'feature_select__model__')
            self._estimator_params[nk] = v

    def _gen_pipeline(self):
        steps = []
        if self.transformation:
            steps.append(('transform', self.transformation_method))
        
        if self.normalisation:
            steps.append(('normalise', self.normalisation_method))

        steps.append(('feature_select', FeatureSelectorSwitcher())) 

        # Speed-up by caching transformation models.
        if self.use_cache: 
            cache_dir = mkdtemp()
        else:
            cache_dir = None
        self.pipe = Pipeline(steps, memory = cache_dir)

        self.pipe_params = {
                **_all_selectors[self.selector],
                **self._cv_params,
                **self._estimator_params,
            }
        self.pipe.set_params(**self.pipe_params)

        
    def run(self):
        X, y = self._prepare_data(self.data)

        with mlflow.start_run(
            run_name = self.selector,
            experiment_id = self.exp_id
        ):
            mlflow.set_tags({'selector': self.selector})
            if self.run_tags is not None:
                mlflow.set_tags(self.run_tags)

            self.pipe.fit(X, y)
            
        feat_selector = self.pipe.named_steps['feature_select']
        return feat_selector  

    
