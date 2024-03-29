import mlflow
import mlflow.sklearn
import random
import pandas as pd
import numpy as np


from mlkit.helpers import mlflow_helpers as mlflow_hp


random.seed = 64

class BaseExperiment():
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
            run_tags = None,
            run_params = None,
            random_state = 64,
    ):
        self._setup_mlflow(mlflow_uri,
                experiment_name,
                experiment_exists_ok,
                run_tags,
                run_params)
        self._setup_data(data,
                target,
                valid_data,
                test_data,
                ignore_features,
                use_features)
        self.target_labels = target_labels
        self.random_state = random_state


    def _setup_mlflow(self, 
            mlflow_uri, 
            experiment_name, 
            experiment_exists_ok,
            run_tags,
            run_params,
    ):
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.sklearn.autolog(log_post_training_metrics=False, silent=True)
        self.exp_id = mlflow_hp.load_experiment(experiment_name, experiment_exists_ok)
        self.run_tags = run_tags
        self.run_params = run_params


    def _setup_data(self,
            data, 
            target,
            valid_data,
            test_data,
            ignore_features,
            use_features,
    ):
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.valid_data = valid_data
        self.test_data = test_data
        if test_data is not None:
            if isinstance(self.test_data, pd.DataFrame): 
                # single test set
                self.test_data.reset_index(drop=True, inplace=True)
            elif isinstance(self.test_data, dict):
                # multiple test sets
                for k, v in self.test_data.items():
                    self.test_data[k] = v.reset_index(drop=True)
            else:
                raise TypeError('test_data must be a pandas DataFrame or a dict of pandas DataFrames')
        
        if valid_data is not None:
            if isinstance(self.valid_data, pd.DataFrame): 
                # single validation set
                self.valid_data.reset_index(drop=True, inplace=True)
            elif isinstance(self.valid_data, dict):
                # multiple validation sets
                for k, v in self.valid_data.items():
                    self.valid_data[k] = v.reset_index(drop=True)
            else:
                raise TypeError('valid_data must be a pandas DataFrame or a dict of pandas DataFrames')

        self.target = target
        self.ignore_features = ignore_features
        self.use_features = use_features

        if use_features is not None:
            self.feature_names = np.array(use_features)
        elif ignore_features is not None:
            self.feature_names = data.loc[:, 
                    ~data.columns.isin(
                        ignore_features+[target]
                    )].columns.to_numpy()
        else:
            self.feature_names = data.loc[:, 
                    data.columns != target].columns.to_numpy()


    def _prepare_data(self, data):
        y = data[self.target]
        X = data.loc[:, self.feature_names]
        return X.to_numpy(), y.to_numpy()


    def eval(self, test_data):
        raise NotImplementedError

    @mlflow_hp.check_active_run
    def run(self):
        raise NotImplementedError


