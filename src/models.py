"""Classifier zoo mirroring the thesis experiments (RBF SVM dropped for speed)."""

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def get_classifiers():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "AdaptiveGB": AdaBoostClassifier(random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
