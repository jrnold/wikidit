"""
Ordinal Classifier
This module contains a Soft Voting/Majority Rule classifier for
classification estimators.
"""

# Authors: Sebastian Raschka <se.raschka@gmail.com>,
#          Gilles Louppe <g.louppe@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import warnings

from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, Ordered
from sklearn.utils import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import Bunch


def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    """Private function used to fit an estimator within a job."""
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


class Ordinallassifier(_BaseComposition, ClassifierMixin, TransformerMixin):
    """Soft/Hard classifier for unfitted estimators of ordered categories."""

    def __init__(self, estimator, voting='hard', weights=None, n_jobs=None,
                 flatten_transform=None):
        self.estimator = estimator
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : pd.Series
            Needs to be a pandas series with type Categorical
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
        Returns
        -------
        self : object
        """
    
        # TODO: move to other stuff
        self.classes_ = list(y.categories)
        self.estimators_ = []
    
        transformed_y = {k: y < k for k in classes_[:-1]}
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, y_cat,
                                                 sample_weight=sample_weight)
                for k in self.classes_[1:])

        #self.named_estimators_ = Bunch(**dict())
        #for k, e in zip(self.classes_[1:], self.estimators_):
        #    self.named_estimators_[k] = e
        return self

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators_')
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self._weights_not_none)),
                axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        check_is_fitted(self, 'estimators_')
        avg = np.average(self._collect_probas(X), axis=0,
                         weights=self._weights_not_none)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns array-like of shape (n_classifiers, n_samples *
                n_classes), being class probabilities calculated by each
                classifier.
            If `voting='soft' and `flatten_transform=False`:
                array-like of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                array-like of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators_')

        if self.voting == 'soft':
            probas = self._collect_probas(X)
            if self.flatten_transform is None:
                warnings.warn("'flatten_transform' default value will be "
                              "changed to True in 0.21. "
                              "To silence this warning you may"
                              " explicitly set flatten_transform=False.",
                              DeprecationWarning)
                return probas
            elif not self.flatten_transform:
                return probas
            else:
                return np.hstack(probas)

        else:
            return self._predict(X)

    def set_params(self, **params):
        """ Setting the parameters for the voting classifier
        Valid parameter keys can be listed with get_params().
        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ``VotingClassifier``,
            the individual classifiers of the ``VotingClassifier`` can also be
            set or replaced by setting them to None.
        Examples
        --------
        # In this example, the RandomForestClassifier is removed
        clf1 = LogisticRegression()
        clf2 = RandomForestClassifier()
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)]
        eclf.set_params(rf=None)
        """
        super(OrdinalClassifier, self)._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        """ Get the parameters of the VotingClassifier
        Parameters
        ----------
        deep : bool
            Setting it to True gets the various classifiers and the parameters
            of the classifiers as well
        """
        return super(OrdinalClassifier, self)._get_params('estimators', deep=deep)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T
