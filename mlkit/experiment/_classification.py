import mlflow
import mlflow.sklearn
import random
import plotly
import inspect
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from tempfile import mkdtemp
from pathlib import Path
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from mlkit.classification.models import _all_clfs
from mlkit.classification.classifier_switcher import ClassifierSwitcher

from ._base import BaseExperiment

random.seed = 64

class ClassificationExperiment(BaseExperiment):
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
            data_splitter = 'sss',
            custom_splitter_kwargs = None,
            stratify_splits = 3,
            stratify_test_size = 0.2,
            group_features = None,
            transformation = False,
            transformation_method = None,
            normalisation = False,
            normalisation_method = None,
            classifiers = 'all',
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
        self._setup_classifiers(classifiers)
        self._setup_data_splitter(data_splitter, 
                group_features,
                stratify_splits,
                stratify_test_size,
                custom_splitter_kwargs)

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
            self._splitter = StratifiedShuffleSplit(n_splits=stratify_splits, 
                                                        test_size=stratify_test_size,
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
        if self.data_splitter == 'sss':
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
        cfm = confusion_matrix(y_true, y_pred)
        auc_ovr = roc_auc_score(y_true, y_proba, multi_class='ovr')
        auc_ovo = roc_auc_score(y_true, y_proba, multi_class='ovo')

        metrics = {
            prefix+'acc': acc,
            prefix+'auc_ovr': auc_ovr,
            prefix+'auc_ovo': auc_ovo,
        }
        mlflow.log_metrics(metrics)
        self.plot_and_log_confusion_matrix(cfm)

        return metrics

    def plot_and_log_confusion_matrix(self, cfm):
        if self.target_labels is not None:
            kwargs = {
                    "x": self.target_labels,
                    "y": self.target_labels,
            }
        else:
            kwargs = {}

        cfm_text = [[str(y) for y in x] for x in cfm]

        fig = ff.create_annotated_heatmap(cfm, annotation_text=cfm_text, colorscale='Dense', **kwargs)
        fig.update_layout(title_text='<b>Confusion matrix</b>')
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))
        fig.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.35,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))

        fig.update_layout(margin=dict(t=50, l=200), width=800, height=600)
        fig['data'][0]['showscale'] = True
        fig['layout']['yaxis']['autorange'] = "reversed"

        mlflow.log_figure(fig, "plots/confusion_matrix.html")
        mlflow.log_dict(plotly.io.to_json(fig), "outputs/_px_confusion_matrix.json")

        cfm_path = Path(self.cache_dir) / 'confusion_matrix.npy'
        np.save(open(cfm_path, 'wb'), cfm) 
        mlflow.log_artifact(cfm_path, "metrics")
        


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

        for clf in tqdm(self.classifiers, desc='Classifiers', leave=False): 
            with mlflow.start_run(
                run_name=clf,
                experiment_id=self.exp_id,
            ) as parent_run:
                mlflow.set_tags({'clf': clf})
                if self.run_tags is not None:
                    mlflow.set_tags(self.run_tags)
                if self.run_params is not None:
                    mlflow.log_params(self.run_params)

                params = _all_clfs[clf]
                self.pipe.set_params(**params) 

                all_metrics = []

                for train_index, validate_index in tqdm(self._gen_splits(X, y), desc='Splits', leave=False):
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





