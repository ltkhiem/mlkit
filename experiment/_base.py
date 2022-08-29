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
            test_data = None,
            use_features = None,
            ignore_features = None,
            mlflow_uri = './',
            experiment_exists_ok = False,
            run_tags = None,
            random_state = 64,
    ):
        self._setup_mlflow(mlflow_uri,
                experiment_name,
                experiment_exists_ok,
                run_tags)
        self._setup_data(data,
                target,
                test_data,
                ignore_features,
                use_features)
        self.random_state = random_state


    def _setup_mlflow(self, 
            mlflow_uri, 
            experiment_name, 
            experiment_exists_ok,
            run_tags
    ):
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.sklearn.autolog(log_post_training_metrics=False)
        self.exp_id = mlflow_hp.load_experiment(experiment_name, experiment_exists_ok)
        self.run_tags = run_tags


    def _setup_data(self,
            data, 
            target,
            test_data,
            ignore_features,
            use_features,
    ):
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.test_data = test_data
        if self.test_data is not None:
            self.test_data.reset_index(drop=True, inplace=True)
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


    @mlflow_hp.check_active_run
    def run(self):
        raise Exception("Abstract method. Not yet implemented")


