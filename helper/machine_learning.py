import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


class MachineLearningEngine:

    def __init__(self, X: pd.DataFrame, y=None):

        self.X = self.preprocessing(X)
        self.y = y
        self.labels = None

    def preprocessing(self, data):
        self.labels = defaultdict(LabelEncoder)
        data = data.apply(lambda x: self.labels[x.name].fit_transform(x) if not np.issubdtype(x.dtype, np.number) else x)
        return data

    def cross_validation_score(self, estimator, n_splits=10, random_state=42):
        cross_validation = StratifiedKFold(n_splits=n_splits, random_state=random_state)

        score = cross_val_score(estimator, self.X, self.y, cv=cross_validation).mean()

        return score

    def find_best_estimator(self):
        # for the moment best estimator is a DecisionTreeClassifier with max_depth=20 and max_features=0.3

        best_estimator = DecisionTreeClassifier(criterion='gini', max_depth=20, max_features=0.3)
        return best_estimator

    def fit_estimator(self, estimator):

        if 'fit' not in dir(estimator):
            raise AttributeError("Estimator must have a fit method")

        estimator.fit(self.X, self.y)
        return estimator

    def predict(self, estimator):
        return estimator.predict(self.X)

