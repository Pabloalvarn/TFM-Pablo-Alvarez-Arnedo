from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_linear_model():
    return LinearRegression()


def get_ridge_model(alpha=1.0):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha))
        ]
    )


def get_lasso_model(alpha=0.01):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=10000))
        ]
    )


def get_random_forest_model(random_state=42):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=random_state
    )


def get_xgboost_model(random_state=42):
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state
    )


def get_lightgbm_model(random_state=42):
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1
    )