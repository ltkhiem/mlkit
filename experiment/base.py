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
        self._setup_mlflow(mlflow_ui,
                experiment_name,
                experiment_exists_ok,
                run_tags)
        self._setupdat(data,
                target,
                test_data,
                ignore_features,
                use_features)
        self.random_state = random_state
        self.use_cache = use_cache


    def _set_up_mlflow(self, 
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


    @mlflow_hp.check_active_run
    def run(self):
        pass


