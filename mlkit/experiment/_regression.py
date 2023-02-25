import inspect
import mlflow
import mlflow.sklearn
import random
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly
from tempfile import mkdtemp
from pathlib import Path


from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

from mlkit.regression.models import _all_regressors
from mlkit.regression.regressor_switcher import RegressorSwitcher

from ._base import BaseExperiment

random.seed = 64

class RegressionExperiment(BaseExperiment):
    def __init__(self, 
            experiment_name,
            data,
            target,
            target_labels = None,
            valid_data = None,
            test_data = None,
            use_features = None,
            ignore_features = None,
            mlflow_uri = './',
            experiment_exists_ok = False,
            data_splitter = 'ss',
            custom_splitter_kwargs = None,
            n_splits = 3,
            test_size = 0.2,
            group_features = None,
            transformation = False,
            transformation_method = None,
            normalisation = False,
            normalisation_method = None,
            regressors = 'all',
            run_tags = None,
            run_params = None,
            random_state = 64,
            use_cache = False
    ):

        super().__init__(experiment_name=experiment_name,
                data=data,
                target=target,
                target_labels=target_labels,
                valid_data=valid_data,
                test_data=test_data,
                use_features=use_features,
                ignore_features=ignore_features,
                mlflow_uri=mlflow_uri,
                experiment_exists_ok=experiment_exists_ok,
                run_tags=run_tags,
                run_params=run_params,
                random_state=random_state)

        self._setup_preprocessing(transformation,
                transformation_method,
                normalisation,
                normalisation_method)
        self._setup_regressors(regressors)
        self._setup_data_splitter(data_splitter, 
                group_features,
                n_splits,
                test_size)

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


    def _setup_regressors(self, regressors):
        if regressors == 'all':
            self.regressors = _all_regressors.keys()
        else:
            self.regressors = regressors 

    def _setup_data_splitter(self, 
            data_splitter, 
            group_features,
            n_splits,
            test_size,
            custom_splitter_kwargs = None
    ):
        self.valid_data_groups = None
        self.test_data_groups = None
        if inspect.isfunction(data_splitter):
            self._splitter = data_splitter
            self.data_splitter = 'custom'
            self.custom_splitter_kwargs = custom_splitter_kwargs
        else:   
            self.data_splitter = data_splitter

        if self.data_splitter == 'sss':
            self._splitter = ShuffleSplit(n_splits=n_splits, 
                                          test_size=test_size,
                                          random_state=self.random_state)
        elif self.data_splitter == 'logo':
            assert group_features is not None, "'groups' has to be specified to use \
                                             Leave One Group Out split."
            self._splitter = LeaveOneGroupOut()
            self.group_features = group_features
            self.data_groups = self.data[group_features].values
            if self.test_data is not None:
                if isinstance(self.test_data, pd.DataFrame):
                    self.test_data_groups = self.test_data[group_features].values
                elif isinstance(self.test_data, dict):
                    self.test_data_groups = {k: v[group_features].values 
                                             for k, v in self.test_data.items()}
            if self.valid_data is not None:
                if isinstance(self.valid_data, pd.DataFrame):
                    self.valid_data_groups = self.valid_data[group_features].values
                elif isinstance(self.valid_data, dict):
                    self.valid_data_groups = {k: v[group_features].values
                                              for k, v in self.valid_data.items()}

        elif self.data_splitter is None:
            self._splitter = lambda x, y: [(None, None)]

    def _gen_splits(self, X, y): 
        if self.data_splitter == 'ss':
            return self._splitter.split(X, y)
        elif self.data_splitter == 'logo':
            return self._splitter.split(X, y, self.data_groups)
        elif self.data_splitter == 'custom':
            # We also send the training data so that custom splitter can have more context
            return self._splitter(self.data, X, y, **self.custom_splitter_kwargs)
        elif self.data_splitter is None:
            return self._splitter(X, y)


    def _gen_pipeline(self):
        steps = []
        if self.transformation:
            steps.append(('transform', self.transformation_method))
        
        if self.normalisation:
            steps.append(('normalise', self.normalisation_method))

        steps.append(('regress', RegressorSwitcher())) 

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

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        corr, p = spearmanr(y_true, y_pred)

        metrics = {
            prefix+'corr': corr,
            prefix+'p_values': p,
            prefix+'rmse': rmse,
            prefix+'r2': r2,
            prefix+'mae': mae,
            prefix+'mse': mse,
        }
        mlflow.log_metrics(metrics)
        return metrics

        
    def gen_run_summary(self, all_metrics):
        df = pd.DataFrame(all_metrics)
        df = df.describe()
        for c in df.columns:
            for x in ['mean', 'std']:
                mlflow.log_metric(f'{c}_{x}', df[c].loc[x])

    def eval(self, 
             X_test, 
             y_test, 
             validate_index=None, 
             test_data_groups=None, 
             prefix=''
     ):
        if self.data_splitter == 'logo':
            _cur_group = self.data_groups[validate_index[0]] 
            _group_idx,  = np.where(test_data_groups==_cur_group)
            return self.eval_and_log_metrics(self.pipe, 
                    X_test[_group_idx], y_test[_group_idx], prefix=f'test_{prefix}')
        else:
            return self.eval_and_log_metrics(self.pipe, 
                    X_test, y_test, prefix=f'test_{prefix}_')

    def run(self):
        X, y = self._prepare_data(self.data)
        if self.test_data is not None:
            if isinstance(self.test_data, pd.DataFrame):
                X_test, y_test = self._prepare_data(self.test_data)
            elif isinstance(self.test_data, dict):
                Xy_test = {k: self._prepare_data(v) for k, v in self.test_data.items()}

        if self.valid_data is not None:
            if isinstance(self.valid_data, pd.DataFrame):
                X_val, y_val = self._prepare_data(self.valid_data)
            elif isinstance(self.valid_data, dict):
                Xy_val = {k: self._prepare_data(v) for k, v in self.valid_data.items()}

        for reg in self.regressors: 
            with mlflow.start_run(
                run_name=reg,
                experiment_id=self.exp_id,
            ) as parent_run:
                mlflow.set_tags({'reg': reg})
                if self.run_tags is not None:
                    mlflow.set_tags(self.run_tags)

                params = _all_regressors[reg]
                self.pipe.set_params(**params) 

                all_metrics = []

                for train_index, validate_index in self._gen_splits(X, y):
                    with mlflow.start_run(
                        experiment_id=self.exp_id,
                        nested=True,
                    ) as child_run:
                        if train_index is not None and validate_index is not None:
                            X_train, X_val = X[train_index], X[validate_index]
                            y_train, y_val = y[train_index], y[validate_index]
                        else:
                            X_train, y_train = X, y

                        self.pipe.fit(X_train, y_train)

                        metrics = {}

                        # Evaluation on validation data
                        if validate_index is not None:
                            metrics.update(self.eval_and_log_metrics(self.pipe, 
                                                    X_val, y_val, prefix='val_'))
                        else:
                            if self.valid_data is not None:
                                if isinstance(self.valid_data, pd.DataFrame):
                                    metrics.update(self.eval(X_val,
                                                             y_val,
                                                             validate_index,
                                                             self.valid_data_groups,
                                                             prefix='valid_'))
                                elif isinstance(self.valid_data, dict):
                                    for k, v in Xy_val.items():
                                        if self.data_splitter=='logo':
                                            metrics.update(self.eval(*v,
                                                                     validate_index,
                                                                     self.valid_data_groups[k],
                                                                     prefix=k))
                                        else:
                                            metrics.update(self.eval(*v, prefix=k))

                        # Evaluation on testing data
                        if self.test_data is not None: 
                            if isinstance(self.test_data, pd.DataFrame):
                                metrics.update(self.eval(X_test, 
                                                         y_test, 
                                                         validate_index, 
                                                         self.test_data_groups,
                                                         prefix='test_'))
                            elif isinstance(self.test_data, dict):
                                for k, v in Xy_test.items():
                                    if self.data_splitter=='logo':
                                        metrics.update(self.eval(*v,
                                                                 validate_index,
                                                                 self.test_data_groups[k],
                                                                 prefix=k))
                                    else: 
                                        metrics.update(self.eval(*v, prefix=k))

                        all_metrics.append(metrics)

                self.gen_run_summary(all_metrics)





