import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def mae(y_true, y_pred):

    return mean_absolute_error(
        y_true,
        y_pred
    )


def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


def smape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (
        np.abs(y_true) + np.abs(y_pred)
    ) / 2

    diff = np.abs(y_true - y_pred)

    return np.mean(
        diff / denominator
    ) * 100