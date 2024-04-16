from sklearn.base import BaseEstimator, RegressorMixin
from numpy.linalg import LinAlgError
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class SafeSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def _create_svr(self):
        return SVR(C=self.C, kernel=self.kernel, gamma=self.gamma)

    def fit(self, X, y):
        self.svr = self._create_svr()
        try:
            self.svr.fit(X, y)
        except LinAlgError:
            print("LinAlgError encountered. Using a default SVR model.")
            self.C = 1.0
            self.kernel = 'rbf'
            self.gamma = 'scale'
            self.svr = self._create_svr()
            self.svr.fit(X, y)
        return self

    def predict(self, X):
        return self.svr.predict(X)

    def get_params(self, deep=True):
        return {'C': self.C, 'kernel': self.kernel, 'gamma': self.gamma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.svr = self._create_svr()
        return self

class SafeRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='auto', bootstrap=True, max_leaf_nodes=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes

    def _create_rf(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42
        )

    def fit(self, X, y):
        self.rf = self._create_rf()
        try:
            self.rf.fit(X, y)
        except LinAlgError:
            print("LinAlgError encountered. Using default RandomForestRegressor model.")
            self.n_estimators = 100
            self.max_depth = None
            self.min_samples_split = 2
            self.min_samples_leaf = 1
            self.max_features = 'auto'
            self.bootstrap = True
            self.max_leaf_nodes = None
            self.rf = self._create_rf()
            self.rf.fit(X, y)
        return self

    def predict(self, X):
        return self.rf.predict(X)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'max_leaf_nodes': self.max_leaf_nodes
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.rf = self._create_rf()
        return self
      
      


class SafeXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1):
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_xgb(self):
        return XGBRegressor(
            objective=self.objective,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate
        )

    def fit(self, X, y):
        self.xgb = self._create_xgb()
        try:
            self.xgb.fit(X, y)
        except LinAlgError:
            print("LinAlgError encountered. Using default XGBRegressor model.")
            self.objective = 'reg:squarederror'
            self.n_estimators = 100
            self.max_depth = 3
            self.learning_rate = 0.1
            self.xgb = self._create_xgb()
            self.xgb.fit(X, y)
        return self

    def predict(self, X):
        return self.xgb.predict(X)

    def get_params(self, deep=True):
        return {
            'objective': self.objective,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.xgb = self._create_xgb()
        return self





class SafeLGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=-1, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _create_lgbm(self):
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate
        )

    def fit(self, X, y):
        self.lgbm = self._create_lgbm()
        try:
            self.lgbm.fit(X, y)
        except LinAlgError:
            print("LinAlgError encountered. Using default LGBMRegressor model.")
            self.n_estimators = 100
            self.max_depth = -1
            self.learning_rate = 0.1
            self.lgbm = self._create_lgbm()
            self.lgbm.fit(X, y)
        return self

    def predict(self, X):
        return self.lgbm.predict(X)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.lgbm = self._create_lgbm()
        return self
