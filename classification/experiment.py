import mlflow
import mlflow.sklearn
import random
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score

from mlkit.classification.models import _all_clfs
from mlkit.classification.classifier_switcher import ClassifierSwitcher


random.seed = 64

class Experiment():
    def __init__(self, 
            experiment_name,
            data,
            target,
            ignore_features = None,
            mlflow_uri = './',
            experiment_exists_ok = False,
            data_splitter = 'sss',
            stratify_splits = 2, 
            stratify_test_size = 0.2,
            group_features = None,
            transformation = False,
            transformation_method = None,
            normalisation = False,
            normalisation_method = None,
            feature_selection = False,
            feature_selection_method = None,
            classifiers = 'all',
            run_tags = None,
            random_state = 64
    ):

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.sklearn.autolog(log_post_training_metrics=False)
        self._load_experiment(experiment_name, experiment_exists_ok)
        self.run_tags = run_tags

        self.data = data
        self.target = target
        self.ignore_features = ignore_features


        self.transformation = transformation
        if self.transformation:
            assert transformation_method, "Please specified transformation method"
            self.transformation_method = transformation_method

        self.normalisation = normalisation
        self.feature_selection = feature_selection


        if classifiers == 'all':
            self.classifiers = _all_clfs.keys()
        else:
            self.classifiers = classifiers 

        self.data_splitter = data_splitter
        if self.data_splitter == 'sss':
            self._splitter = StratifiedShuffleSplit(n_splits=stratify_splits, 
                                                        test_size=stratify_test_size,
                                                        random_state=random_state)
        elif self.data_splitter == 'logo':
            assert group_features is not None, "'groups' has to be specified to use \
                                             Leave One Group Out split."
            self._splitter = LeaveOneGroupOut()
            self.group_features = group_features
            self.data_groups = data[group_features].values



        self.random_state = random_state

        self._gen_pipeline() 

    def _load_experiment(self, experiment_name, exp_exists_ok):
        expm = mlflow.get_experiment_by_name(experiment_name)
        if expm is not None:
            if exp_exists_ok:
                self.exp_id = dict(expm)['experiment_id']
                return
            else:
                confirm = input(f"Experiment {experiment_name} exists. Press Enter to continue, type 'quit' to abort operation.\n")
                if confirm == '': 
                    self.exp_id = dict(expm)['experiment_id']
                    return
                raise Exception("Experiment exists")

        self.exp_id = mlflow.create_experiment(experiment_name)

    def _prepare_data(self):
        y = self.data[self.target]
        X = self.data.loc[:, self.data.columns != self.target]
        if self.ignore_features is not None:
            X = X.loc[:, ~X.columns.isin(self.ignore_features)]
        return X.to_numpy(), y.to_numpy()


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

        if self.feature_selection:
            steps.append(('feat_select', self.feature_selection_method))

        steps.append(('classify', ClassifierSwitcher())) 

        self.pipe = Pipeline(steps) 
        

    def _check_active_run(self):
        """
        Have to implement this properly
        """
        mlflow.end_run()


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
        self._check_active_run()
        X, y = self._prepare_data()

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
                        metrics = self.eval_and_log_metrics(self.pipe, 
                                X_val, y_val, prefix='val_')
                        all_metrics.append(metrics)

                self.gen_run_summary(all_metrics)





         


