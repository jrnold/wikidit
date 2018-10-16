from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.utils import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import Bunch

import numpy as np
import pandas as pd

def _parallel_fit_estimator(estimator, X, y, cat):
    touse = y >= cat
    y_transform = y > cat
    return estimator.fit(X[touse, :], y_transform[touse])

class SequentialClassifier(_BaseComposition, ClassifierMixin, TransformerMixin):

    def __init__(self, estimator, n_jobs=None, proba_transform=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.proba_transform = proba_transform

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimator))

    def fit(self, X, y, categories='auto'):
        # this is hard-coded for categorical variables
        self.classes_ = y.categories

        categories = self.classes_[:-1]

        # order of estimators
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(self.estimator),
                                                 X, y, cat)
                for cat in categories)

        self.named_estimators_ = Bunch(**dict())
        for k, e in zip(self.classes_[:-1], self.estimators_):
            self.named_estimators_[k] = e
        return self

    def predict(self, X):
        # For prediction use the median class, not the modal class
        cdf = np.cumsum(self.predict_proba(X), axis=1)
        out = np.argmax(cdf >= 0.5, axis=1)
        out = pd.Categorical.from_codes(out, categories=self.classes_,
                                        ordered=True)
        return out

    def _collect_log_probas(self, X):
        """Collect results from predict calls. """
        # If it has log_proba available, use it since we will be multiplying
        # by probabilities.
        if hasattr(self.estimator, "predict_log_proba"):
            return [clf.predict_log_proba(X) for clf in self.estimators_]
        else:
            return [np.log(clf.predict_proba(X)) for clf in self.estimators_]

    def _predict_log_proba(self, X):
        """Predict log class probabilities for X"""
        out = np.empty((X.shape[0], len(self.classes_)))
        for i, logp in enumerate(self._collect_log_probas(X)):
            if i > 0:
                # add log conditional probability
                logp += out[:, (i, )]
            out[:, i:(i + 2)] = logp
        return out

    @property
    def predict_log_proba(self):
        return self._predict_log_proba

    def predict_proba(self, X):
        # work with on log scale as long as possible since the probabilities
        # are being multiplied
        return np.exp(self.predict_log_proba(X))

    def transform(self, X):
        if self.proba_transform:
            return self.predict_proba(X)
        else:
            return self.predict(X)

    def get_params(self, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        if hasattr(self.estimator, 'get_params'):
            for key, value in self.estimator.get_params(deep=True).items():
                out['estimator__%s' % key] = value
        return out

    def set_params(self, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if 'estimator' in params:
            setattr(self, 'estimator', params.pop(attr))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def score(self, X, y):
        return 1 - (np.mean(np.abs(self.predict(X) - y.codes)) / (len(self.classes_) - 1))