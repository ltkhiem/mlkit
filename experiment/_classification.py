import mlflow
import mlflow.sklearn
import random
import pandas as pd
import numpy as np
from tempfile import mkdtemp

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score

from mlkit.classification.models import _all_clfs
from mlkit.classification.classifier_switcher import ClassifierSwitcher

from ._base import BaseExperiment

random.seed = 64

class ClassificationExperiment(BaseExperiment):
    def __init__(self, 
            experiment_name,
            data,
            target,
            test_data = None,
            use_features = None,
            ignore_features = None,
            mlflow_uri = './',
            experiment_exists_ok = False,
            data_splitter = 'sss',
            stratify_splits = 3,
            stratify_test_size = 0.2,
            group_features = None,
            transformation = False,
            transformation_method = None,
            normalisation = False,
            normalisation_method = None,
            classifiers = 'all',
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
        self._setup_classifiers(classifiers)
        self._setup_data_splitter(data_splitter, 
                group_features,
                stratify_splits,
                stratify_test_size)

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


    def _setup_classifiers(self, classifiers):
        if classifiers == 'all':
            self.classifiers = _all_clfs.keys()
        else:
            self.classifiers = classifiers 


    def _setup_data_splitter(self, 
            data_splitter, 
            group_features,
            stratify_splits,
            stratify_test_size,
    ):
        self.data_splitter = data_splitter
        if self.data_splitter == 'sss':
            self._splitter = StratifiedShuffleSplit(n_splits=stratify_splits, 
                                                        test_size=stratify_test_size,
                                                        random_state=self.random_state)
        elif self.data_splitter == 'logo':
            assert group_features is not None, "'groups' has to be specified to use \
                                             Leave One Group Out split."
            self._splitter = LeaveOneGroupOut()
            self.group_features = group_features
            self.data_groups = data[group_features].values
            if self.test_data is not None:
                self.test_data_groups = test_data[group_features].values


    def _gen_splits(self, X, y): 
        if self.data_splitter == 'sss':
            return self._splitter.split(X, y)
        elif self.data_splitter == 'logo':
            return self._splitter.split(X, y, self.data_groups)


    def _gen_pipeline(self):
        steps = []
        if self.transformation:
            steps.append(('transform', self.transformation_method))
        
        if self.normalisation:
            steps.append(('normalise', self.normalisation_method))

        steps.append(('classify', ClassifierSwitcher())) 

        # Speed-up by caching transformation models.
        self.cache_dir = mkdtemp()
        kwargs = {"memory":self.cache_dir} if self.use_cache else {}
        self.pipe = Pipeline(steps, **kwargs)
        

    def log_train_val_index(self, train_index, test_index):
        mlflow.log_dict({
            "train_index": train_index,
            "val_index": test_index,
        }, "outputs/train_val_index.json")


    def eval_and_log_metrics(self, model, X, y_true, prefix):
        y_pred = model.predict(X)
        try:
            y_proba = model.predict_proba(X)
        except:
            d = model.decision_function(X)
            y_proba = np.array([np.exp(d[x])/np.sum(np.exp(d[x]))
                for x in range(d.shape[0])])
            if np.isnan(y_proba).any():
                y_proba = np.nan_to_num(y_proba, nan=0, posinf=0)
                _idx, = np.where(y_proba.sum(axis=1) != 1)
                _size = y_proba[0].shape[0]
                for i in _idx:
                    y_proba[i] = np.full((1, _size), 1/_size)


        acc = accuracy_score(y_true, y_pred)
        auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr')
        auc_ovo = roc_auc_score(y_true, y_proba, multi_class='ovo')

        metrics = {
            prefix+'acc': acc,
            prefix+'auc_ovr': auc_ovr,
            prefix+'auc_ovo': auc_ovo,
        }
        mlflow.log_metrics(metrics)
        return metrics


    def gen_run_summary(self, all_metrics):
        df = pd.DataFrame(all_metrics)
        df = df.describe()
        for c in df.columns:
            for x in ['mean', 'std']:
                mlflow.log_metric(f'{c}_{x}', df[c].loc[x])


    def run(self):
        X, y = self._prepare_data(self.data)
        if self.test_data is not None:
            X_test, y_test = self._prepare_data(self.test_data)

        for clf in self.classifiers: 
            with mlflow.start_run(
                run_name=clf,
                experiment_id=self.exp_id,
            ) as parent_run:
                mlflow.set_tags({'clf': clf})
                if self.run_tags is not None:
                    mlflow.set_tags(self.run_tags)

                params = _all_clfs[clf]
                self.pipe.set_params(**params) 

                all_metrics = []

                for train_index, validate_index in self._gen_splits(X, y):
                    with mlflow.start_run(
                        experiment_id=self.exp_id,
                        nested=True,
                    ) as child_run:
                        X_train, X_val = X[train_index], X[validate_index]
                        y_train, y_val = y[train_index], y[validate_index]

                        self.pipe.fit(X_train, y_train)


                        # Evaluation on validation data
                        metrics = self.eval_and_log_metrics(self.pipe, 
                                X_val, y_val, prefix='val_')

                        # Evaluation on testing data
                        if self.test_data is not None: 
                            if self.data_splitter == 'logo':
                                _cur_group = self.data_groups[validate_index[0]] 
                                _group_idx,  = np.where(self.test_data_groups==_cur_group)
                                metrics.update(self.eval_and_log_metrics(self.pipe, 
                                        X_test[_group_idx], y_test[_group_idx], prefix='test_'))
                            else:
                                metrics.update(self.eval_and_log_metrics(self.pipe, 
                                        X_test, y_test, prefix='test_'))


                        all_metrics.append(metrics)

                self.gen_run_summary(all_metrics)





