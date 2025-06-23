# dummy_model.py

import random
from sklearn.base import BaseEstimator

class DummyASLModel(BaseEstimator):
    def predict(self, X):
        return [random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) for _ in range(len(X))]
