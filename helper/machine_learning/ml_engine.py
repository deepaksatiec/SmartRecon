import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline

from helper.encoding import OneHotEncoder, MultiColumnLabelEncoder
from helper.helper import just_transforms


AVAILABLE_ALGO = ['knn', 'random_forest', 'decision_tree', 'tpot']

# Decision tree params
MAX_DEPTH = 20

# knn params
MINIMUM_FRACTIONS = 0.01
N_NEIGHBORS = 1
WEIGHTS = 'uniform'
P = 2  # Power parameter for the Minkowski metric


class MachineLearningEngine:
    """
    Engine with machine learning capabilities.

    """

    def __init__(self, X: pd.DataFrame, feat_types, y=None):

        if y is not None:
            X_rows = X.shape[0]
            y_rows = y.shape[0]
            if not X_rows == y_rows:
                raise ValueError("X and y must have the same row number. X has {x_row} rows but y has " \
                                 "{y_row}".format(x_row=X_rows, y_row=y_rows))
        if y is not None:
            self.y = y.reset_index(drop=True)
        else:
            self.y = y

        self.X = X.reset_index(drop=True)
        self.feat_types = feat_types

        self._cate_feat = None
        self._LabelEncoder = None
        self._OneHotEncoder = None
        self._pipeline = None
        self._MaxAbsScaler = None

    @property
    def cate_feat(self):
        if self._cate_feat is None:
            categorical_feat = [feature for feature in self.X.columns if
                                (self.feat_types[feature].lower() == 'categorical')]
            self._cate_feat = categorical_feat
        return self._cate_feat

    @property
    def LabelEncoder(self):
        if self._LabelEncoder is None:
            self._LabelEncoder = MultiColumnLabelEncoder(columns=self.cate_feat)
        return self._LabelEncoder

    @property
    def OneHotEncoder(self):
        if self._OneHotEncoder is None:
            self._OneHotEncoder = OneHotEncoder(minimum_fraction=MINIMUM_FRACTIONS,
                                                categorical_features=[(column in self.cate_feat) for column in
                                                                      self.X.columns],
                                                sparse=False)

        return self._OneHotEncoder

    @property
    def MaxAbsScaler(self):
        if self._MaxAbsScaler is None:
            self._MaxAbsScaler = MaxAbsScaler(copy=False)
        return self._MaxAbsScaler

    def pipeline(self, algo):
        """
        A custom pipeline for demo purpose.

        :param algo: str, either 'knn',  'decision_tree' or 'tpot'.
        :return: a sklearn.pipeline.Pipeline object.
        """

        if algo not in AVAILABLE_ALGO:
            raise ValueError('{0} wrong algo. Must be either {1}.'.format(algo, ' ,'.join(AVAILABLE_ALGO)))

        # classification estimator
        if np.issubdtype(self.y.dtype, np.integer):

            # one NN - Slow to fit but easy to analyse. - Possible accuracy improvement with polynomial transformation
            if algo == 'knn':
                pipeline = Pipeline(steps=[
                    ('label_encoder', self.LabelEncoder),
                    ('one_hot_encoder', self.OneHotEncoder),
                    ('standardize', self.MaxAbsScaler),
                    ('knn', KNeighborsClassifier(n_neighbors=N_NEIGHBORS, p=P, weights=WEIGHTS, n_jobs=-1))
                ])

            # decision tree - Fast and good accuracy - Can be hard to analyse.
            elif algo == 'decision_tree':
                pipeline = Pipeline(steps=[
                    ('label_encoder', self.LabelEncoder),
                    ('decision_tree', DecisionTreeClassifier(max_depth=MAX_DEPTH))
                ])
            # tpot
            elif algo == 'tpot':
                pipeline = TPOTClassifier(generations=5, population_size=50, verbosity=2)

        # default estimator
        else:
            pipeline = Pipeline(steps=[
                ('label_encoder', self.LabelEncoder),
                ('decision_tree', DecisionTreeClassifier(max_depth=MAX_DEPTH))
            ])

        return pipeline

    def cv_metrics(self, estimator, random_state=42, test_size=0.2, normalize=False):
        """
        A dict of score/ metrics of the estimator performance on the test-set.

        :param estimator: any estimator
        :param random_state: int, default 42
        :param test_size: float, default 0.2
        :param normalize: boolean, default False. Normalize the confusion matrix.
        :return: dict composed of accuracy_score, precision_score, recall_score, confusion_matrix
        """
        X_train_, X_test_, y_train_, y_test_ = train_test_split(self.X, self.y, test_size=test_size,
                                                                random_state=random_state)
        fitted_estimator = estimator.fit(X=X_train_, y=y_train_)
        y_pred = fitted_estimator.predict(X_test_)

        metrics = {
            metric.__name__: metric.__call__(y_test_, y_pred)
            for metric in [accuracy_score, precision_score, recall_score, confusion_matrix]
        }

        if normalize:
            cm = metrics[confusion_matrix.__name__]
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, 2)
            metrics[confusion_matrix.__name__] = cm

        metrics[confusion_matrix.__name__] = metrics[confusion_matrix.__name__].tolist()
        return metrics

    def k_folds_score(self, estimator, n_splits=5, random_state=42):
        cross_validation = StratifiedKFold(n_splits=n_splits, random_state=random_state)

        score = cross_val_score(estimator, self.X, self.y, cv=cross_validation).mean()

        return score

    def fit_estimator(self, estimator):

        if not hasattr(estimator, 'fit'):
            raise AttributeError("Estimator must have a fit method")

        estimator.fit(self.X, self.y)
        return estimator

    def predict(self, estimator):
        return estimator.predict(self.X)

    def kneighbors(self, estimator):
        """
        If estimator is a knn or a pipeline with a knn as final step, finds the K-neighbors of a point.
        :param estimator:
        :return: array, Indices of the nearest points in the population matrix.
        """
        result = None

        if isinstance(estimator, Pipeline):
            lst_estimator = estimator.steps[-1][-1]

            if isinstance(lst_estimator, KNeighborsClassifier):
                result = lst_estimator.kneighbors(just_transforms(estimator, self.X), return_distance=False)

        elif isinstance(estimator, KNeighborsClassifier):
            result = estimator.kneighbors(self.X, return_distance=False)

        if result is None:
            raise ValueError("Estimator must be a knn or a Pipeline with knn in final place.")

        return result

