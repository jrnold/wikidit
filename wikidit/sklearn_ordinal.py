from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import Bunch


class OrdinalClassifier(_BaseComposition, ClassifierMixin, TransformerMixin):

    def __init__(self, estimator, n_jobs=None, proba_transform=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.proba_transform = proba_transform

    @property
    def named_estimators(self):
        return Bunch(**dict(self.estimator))

    def fit(self, X, y, categories='auto'):
        if not (isinstance(y, pd.Series) and hasattr(y, "cat")):
            raise ValueError("y must be pd.Series object with dtype Categorical")

        # this is hard-coded for categorical variables
        self.classes_ = y.cat.categories

        categories = self.classes_[:-1]
    
        # order of estimators
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(self.estimator), X, y, cat)
                for cat in categories)

        self.named_estimators_ = Bunch(**dict())
        for k, e in zip(self.classes_[:-1], self.estimators_):
            self.named_estimators_[k] = e
        return self

    def predict(self, X):
        out = np.argmax(self.predict_proba(X), axis=1)
        out = pd.Categorical.from_codes(out, categories=self.classes_, ordered=True)
        return out

    def _collect_log_probas(self, X):
        """Collect results from clf.predict calls. """
        # If it has log_proba available, use it since we will be multiplying 
        # by probabilities.
        if hasattr(clf, "predict_log_proba"):
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
    
    def set_params(self, **params):
        super(OrdinalClassifier, self)._set_params('estimator', **params)
        return self

    def get_params(self, deep=True):
        return super(OrdinalClassifier, self)._get_params('estimator', deep=deep)
