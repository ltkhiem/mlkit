import mlflow
import mlflow.sklearn
import random
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from constants import ALL_CLASSIFIERS


random.seed = 64

class Experiment():
    def __init__(self, 
            experiment_name,
            data,
            target,
            ignore_features = None,
            mlflow_uri = './'
            data_splitter = 'sss',
            stratify_features = None,
            stratify_splits = 10, 
            stratify_test_size = 0.2,
            group_features = None
            transformation = False,
            transformation_method = None,
            normalisation = False,
            normalisation_method = None,
            feature_selection = False,
            feature_selection_method = None,
            classifiers = 'all'
            random_state = 64
    ):

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.sklearn.autolog()
        self.exp_id = mlflow.create_experiment(experiment_name)

        self.transformation = transformation
        self.normalisation = normalisation

        
        if classifiers == 'all':
            self.classifiers = ALL_CLASSIFIERS
        else:
            assert type(classifiers)==list, "'classifiers' should be a list"
            self.classifiers = classifiers


        self.data_splitter = data_splitter
        if self.data_splitter = 'sss':
            if stratify_features is None:
                stratify_features = target
            self._splitter = StratifiedShuffleSplit(n_splits=stratify_splits, 
                                                        test_size=strafify_test_size,
                                                        random_state=random_state)
        elif self.data_splitter = 'logo':
            assert group_features is not None, "'groups' has to be specified to use \
                                             Leave One Group Out split."
            self._splitter = LeaveOneGroupOut()
            self.group_features = group_features


        self.random_state = random_state


    def _gen_splits(self, X, y): 
        if self.data_splitter = 'sss':
            return self._splitter.split(X, y)
        elif data_splitter = 'logo':
            return self._splitter.split(X, y, self.group_features)


    def _gen_pipeline(self):
        steps = []
        if self.transformation:
            steps.append(('transform', self.transformation_method))
        
        if self.normalisation:
            steps.append(('normalise', self.normalisation_method))

        if self.feature_selection:
            steps.append(('feat_select', self.feature_selection_method))

        steps.append(('classify': ClassifierSwicher())) 

        self.pipe = Pipeline(steps) 
        

    def run():
        
        for train_index, validate_index in self._gen_splits(X, y):
            with mlflow.start_run(experiment_id = self.exp_id):
                X_train, X_val = X[train_index], X[validate_index]
                y_train, y_val = y[train_index], y[validate_index]

            self.pipe.fit(X_train, y_train)
            y_pred = self.pipe.predict(X_val, y_val)
        


