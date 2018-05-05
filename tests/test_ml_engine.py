from sklearn.model_selection import train_test_split
from helper.machine_learning import MachineLearningEngine


def test_MachineLearningEngine(data, feat_types):
    X, y = data[['Item', 'GOP', 'Family', 'Before Amount']], data['Action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)

    engine = MachineLearningEngine(X=X_train, y=y_train, feat_types=feat_types)
    pipe = engine.pipeline

    assert engine.cross_validation_score(estimator=pipe, random_state=None) > 0.5
