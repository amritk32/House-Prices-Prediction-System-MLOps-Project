from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def get_xgboost_model():
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )


def get_ridge_model():
    return RidgeCV(alphas=[0.1, 1.0, 5.0, 10.0, 50.0, 100.0], cv=5)


def get_lgbm_model():
    return LGBMRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=20,
        learning_rate=0.1,
        num_leaves=31,
    )


def get_voting_ensemble(estimators, weights=None):
    return VotingRegressor(estimators=estimators, weights=weights)


def train_model(estimator, X_train, y_train):
    estimator.fit(X_train, y_train)
    return estimator
